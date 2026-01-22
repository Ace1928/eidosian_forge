import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config
from flash_attn.models.bigcode import remap_state_dict_hf_bigcode
from flash_attn.models.falcon import remap_state_dict_hf_falcon
from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox
from flash_attn.models.gptj import remap_state_dict_hf_gptj
from flash_attn.models.llama import remap_state_dict_hf_llama
from flash_attn.models.opt import remap_state_dict_hf_opt
from flash_attn.modules.block import Block, ParallelBlock
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import (
from flash_attn.ops.activations import sqrelu_fwd
from flash_attn.utils.distributed import (
from flash_attn.utils.generation import GenerationMixin
from flash_attn.utils.pretrained import state_dict_from_pretrained
class GPTModel(GPTPreTrainedModel):

    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None):
        super().__init__(config)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.process_group = process_group
        self.sequence_parallel = getattr(config, 'sequence_parallel', True)
        assert config.activation_function in ['gelu', 'gelu_new', 'gelu_fast', 'gelu_approx', 'gelu_pytorch_tanh', 'relu', 'sqrelu', 'glu', 'swiglu', 'geglu']
        pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
        vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        self.embeddings_multiplier = getattr(config, 'mup_embeddings_multiplier', 1.0)
        self.residual_in_fp32 = getattr(config, 'residual_in_fp32', False)
        self.prenorm = getattr(config, 'prenorm', True)
        use_rms_norm = getattr(config, 'rms_norm', False)
        word_embed_proj_dim = getattr(config, 'word_embed_proj_dim', None)
        self.parallel_block = getattr(config, 'parallel_block', False)
        if process_group is None:
            self.embeddings = GPT2Embeddings(config.hidden_size, vocab_size, config.max_position_embeddings, word_embed_proj_dim=word_embed_proj_dim, **factory_kwargs)
        else:
            self.embeddings = ParallelGPT2Embeddings(config.hidden_size, vocab_size, config.max_position_embeddings, process_group=process_group, sequence_parallel=self.sequence_parallel, **factory_kwargs)
        self.layers = nn.ModuleList([create_block(config, layer_idx=i, process_group=process_group, **factory_kwargs) for i in range(config.num_hidden_layers)])
        rotary_emb_fraction = getattr(config, 'rotary_emb_fraction', 0.0)
        if rotary_emb_fraction > 0.0:
            for layer in self.layers[1:]:
                layer.mixer.rotary_emb = self.layers[0].mixer.rotary_emb
        self.fused_dropout_add_ln = getattr(config, 'fused_dropout_add_ln', False)
        if self.fused_dropout_add_ln:
            if layer_norm_fn is None:
                raise ImportError('Triton is not installed')
        if self.prenorm:
            self.drop_f = nn.Dropout(config.resid_pdrop)
            norm_cls = nn.LayerNorm if not use_rms_norm else RMSNorm
            self.ln_f = norm_cls(config.hidden_size, eps=config.layer_norm_epsilon, **factory_kwargs)
        if process_group is not None:
            for p in self.ln_f.parameters():
                p._shared_params = True
                if self.sequence_parallel:
                    p._sequence_parallel = True
        self.apply(partial(_init_weights, n_layer=config.num_hidden_layers, initializer_range=config.initializer_range, mup_width_scale=getattr(config, 'mup_width_scale', 1.0)))
        self.tie_weights()

    def tie_weights(self):
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs) for i, layer in enumerate(self.layers)}

    def forward(self, input_ids, position_ids=None, inference_params=None):
        embedding_kwargs = {'combine_batch_seqlen_dim': True} if self.process_group is not None and self.sequence_parallel else {}
        hidden_states = self.embeddings(input_ids, position_ids=position_ids, **embedding_kwargs)
        if self.embeddings_multiplier != 1.0:
            hidden_states = hidden_states * self.embeddings_multiplier
        if self.parallel_block:
            hidden_states2 = None
        residual = None
        mixer_kwargs = {'seqlen': input_ids.shape[1]} if self.process_group is not None and self.sequence_parallel else {}
        if inference_params is not None:
            mixer_kwargs['inference_params'] = inference_params
        for layer in self.layers:
            if self.prenorm:
                if not self.parallel_block:
                    hidden_states, residual = layer(hidden_states, residual, mixer_kwargs=mixer_kwargs)
                else:
                    hidden_states, hidden_states2, residual = layer(hidden_states, hidden_states2, residual, mixer_kwargs=mixer_kwargs)
            else:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_f(hidden_states)
                if not self.parallel_block:
                    residual = dropped + residual if residual is not None else dropped
                else:
                    dropped2 = self.drop_f(hidden_states2)
                    residual = residual + dropped + dropped2 if residual is not None else dropped + dropped2
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                hidden_states = layer_norm_fn(hidden_states, self.ln_f.weight, self.ln_f.bias, residual=residual, x1=None if not self.parallel_block else hidden_states2, eps=self.ln_f.eps, dropout_p=self.drop_f.p if self.training else 0.0, prenorm=False, is_rms_norm=isinstance(self.ln_f, RMSNorm))
        return hidden_states