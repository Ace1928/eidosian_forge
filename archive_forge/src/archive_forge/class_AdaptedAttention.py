import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import TRANSFORMERS_MODEL_CONFIG
class AdaptedAttention(nn.Module):
    """This module wraps a LLamaAttention module and injects adaption prompts."""

    def __init__(self, model_type: str, adapter_len: int, model):
        """
        Initialize object.

        Args:
            model_type: The transformer model type. This is used to retrieve the right method to
                compute query states.
            adapter_len: The length of the adaption prompt to insert.
            model: The original transformer attention module that is being wrapped.
        """
        assert not isinstance(model, AdaptedAttention)
        super().__init__()
        self.model_type = model_type
        self.model = model
        self.adapter_len = adapter_len
        device = next(model.parameters()).device
        target_dtype = model.q_proj.weight.dtype if model.q_proj.weight.dtype not in [torch.int8, torch.uint8] else torch.float32
        self.adaption_prompt = nn.Parameter(torch.empty(1, adapter_len, self.model.hidden_size, device=device, dtype=target_dtype).normal_())
        self.adaption_gate = nn.Parameter(torch.zeros(1, device=device, dtype=target_dtype))

    def forward(self, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.

        "Official" paper implementation:
        https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L141

        Args:
            kwargs: See the original LlamaAttention module.
        """
        if kwargs.get('output_attention', False):
            raise NotImplementedError('output_attention is not currently supported.')
        output, _, past_key_value = self.model(**kwargs)
        bsz = output.shape[0]
        q_len = output.shape[1]
        embed_dim = output.shape[2]
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer
        if k_proj_layer == v_proj_layer:
            _, key, value = getattr(self.model, k_proj_layer)(self.adaption_prompt).split(embed_dim, dim=2)
        else:
            key = getattr(self.model, k_proj_layer)(self.adaption_prompt)
            value = getattr(self.model, v_proj_layer)(self.adaption_prompt)
        adapter_k = key.view(1, self.adapter_len, self.model.num_heads, self.model.head_dim).repeat(bsz, 1, 1, 1).transpose(1, 2)
        adapter_v = value.view(1, self.adapter_len, self.model.num_heads, self.model.head_dim).repeat(bsz, 1, 1, 1).transpose(1, 2)
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        query_states = compute_query_states(model=self.model, **kwargs)
        previous_dtype = query_states.dtype
        scores = torch.matmul(query_states, adapter_k.transpose(2, 3).to(previous_dtype)) / math.sqrt(self.model.head_dim)
        scores = self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32).to(previous_dtype)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)
        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)
        output = output + adapter_output
        output = output.to(previous_dtype)
        return (output, None, past_key_value)