import math
from typing import List, Optional, Tuple
import torch
class _EmformerAttention(torch.nn.Module):
    """Emformer layer attention module.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Emformer layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        weight_init_gain (float or None, optional): scale factor to apply when initializing
            attention module parameters. (Default: ``None``)
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(self, input_dim: int, num_heads: int, dropout: float=0.0, weight_init_gain: Optional[float]=None, tanh_on_mem: bool=False, negative_inf: float=-100000000.0):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(f'input_dim ({input_dim}) is not a multiple of num_heads ({num_heads}).')
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.tanh_on_mem = tanh_on_mem
        self.negative_inf = negative_inf
        self.scaling = (self.input_dim // self.num_heads) ** (-0.5)
        self.emb_to_key_value = torch.nn.Linear(input_dim, 2 * input_dim, bias=True)
        self.emb_to_query = torch.nn.Linear(input_dim, input_dim, bias=True)
        self.out_proj = torch.nn.Linear(input_dim, input_dim, bias=True)
        if weight_init_gain:
            torch.nn.init.xavier_uniform_(self.emb_to_key_value.weight, gain=weight_init_gain)
            torch.nn.init.xavier_uniform_(self.emb_to_query.weight, gain=weight_init_gain)

    def _gen_key_value(self, input: torch.Tensor, mems: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T, _, _ = input.shape
        summary_length = mems.size(0) + 1
        right_ctx_utterance_block = input[:T - summary_length]
        mems_right_ctx_utterance_block = torch.cat([mems, right_ctx_utterance_block])
        key, value = self.emb_to_key_value(mems_right_ctx_utterance_block).chunk(chunks=2, dim=2)
        return (key, value)

    def _gen_attention_probs(self, attention_weights: torch.Tensor, attention_mask: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attention_weights_float = attention_weights.float()
        attention_weights_float = attention_weights_float.masked_fill(attention_mask.unsqueeze(0), self.negative_inf)
        T = attention_weights.size(1)
        B = attention_weights.size(0) // self.num_heads
        if padding_mask is not None:
            attention_weights_float = attention_weights_float.view(B, self.num_heads, T, -1)
            attention_weights_float = attention_weights_float.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), self.negative_inf)
            attention_weights_float = attention_weights_float.view(B * self.num_heads, T, -1)
        attention_probs = torch.nn.functional.softmax(attention_weights_float, dim=-1).type_as(attention_weights)
        return torch.nn.functional.dropout(attention_probs, p=float(self.dropout), training=self.training)

    def _forward_impl(self, utterance: torch.Tensor, lengths: torch.Tensor, right_context: torch.Tensor, summary: torch.Tensor, mems: torch.Tensor, attention_mask: torch.Tensor, left_context_key: Optional[torch.Tensor]=None, left_context_val: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = utterance.size(1)
        T = right_context.size(0) + utterance.size(0) + summary.size(0)
        query = self.emb_to_query(torch.cat([right_context, utterance, summary]))
        key, value = self.emb_to_key_value(torch.cat([mems, right_context, utterance])).chunk(chunks=2, dim=2)
        if left_context_key is not None and left_context_val is not None:
            right_context_blocks_length = T - torch.max(lengths).int() - summary.size(0)
            key = torch.cat([key[:mems.size(0) + right_context_blocks_length], left_context_key, key[mems.size(0) + right_context_blocks_length:]])
            value = torch.cat([value[:mems.size(0) + right_context_blocks_length], left_context_val, value[mems.size(0) + right_context_blocks_length:]])
        reshaped_query, reshaped_key, reshaped_value = [tensor.contiguous().view(-1, B * self.num_heads, self.input_dim // self.num_heads).transpose(0, 1) for tensor in [query, key, value]]
        attention_weights = torch.bmm(reshaped_query * self.scaling, reshaped_key.transpose(1, 2))
        padding_mask = _gen_padding_mask(utterance, right_context, summary, lengths, mems, left_context_key)
        attention_probs = self._gen_attention_probs(attention_weights, attention_mask, padding_mask)
        attention = torch.bmm(attention_probs, reshaped_value)
        if attention.shape != (B * self.num_heads, T, self.input_dim // self.num_heads):
            raise AssertionError('Computed attention has incorrect dimensions')
        attention = attention.transpose(0, 1).contiguous().view(T, B, self.input_dim)
        output_right_context_mems = self.out_proj(attention)
        summary_length = summary.size(0)
        output_right_context = output_right_context_mems[:T - summary_length]
        output_mems = output_right_context_mems[T - summary_length:]
        if self.tanh_on_mem:
            output_mems = torch.tanh(output_mems)
        else:
            output_mems = torch.clamp(output_mems, min=-10, max=10)
        return (output_right_context, output_mems, key, value)

    def forward(self, utterance: torch.Tensor, lengths: torch.Tensor, right_context: torch.Tensor, summary: torch.Tensor, mems: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        B: batch size;
        D: feature dimension of each frame;
        T: number of utterance frames;
        R: number of right context frames;
        S: number of summary elements;
        M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape `(T, B, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``utterance``.
            right_context (torch.Tensor): right context frames, with shape `(R, B, D)`.
            summary (torch.Tensor): summary elements, with shape `(S, B, D)`.
            mems (torch.Tensor): memory elements, with shape `(M, B, D)`.
            attention_mask (torch.Tensor): attention mask for underlying attention module.

        Returns:
            (Tensor, Tensor):
                Tensor
                    output frames corresponding to utterance and right_context, with shape `(T + R, B, D)`.
                Tensor
                    updated memory elements, with shape `(M, B, D)`.
        """
        output, output_mems, _, _ = self._forward_impl(utterance, lengths, right_context, summary, mems, attention_mask)
        return (output, output_mems[:-1])

    @torch.jit.export
    def infer(self, utterance: torch.Tensor, lengths: torch.Tensor, right_context: torch.Tensor, summary: torch.Tensor, mems: torch.Tensor, left_context_key: torch.Tensor, left_context_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for inference.

        B: batch size;
        D: feature dimension of each frame;
        T: number of utterance frames;
        R: number of right context frames;
        S: number of summary elements;
        M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape `(T, B, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``utterance``.
            right_context (torch.Tensor): right context frames, with shape `(R, B, D)`.
            summary (torch.Tensor): summary elements, with shape `(S, B, D)`.
            mems (torch.Tensor): memory elements, with shape `(M, B, D)`.
            left_context_key (torch.Tensor): left context attention key computed from preceding invocation.
            left_context_val (torch.Tensor): left context attention value computed from preceding invocation.

        Returns:
            (Tensor, Tensor, Tensor, and Tensor):
                Tensor
                    output frames corresponding to utterance and right_context, with shape `(T + R, B, D)`.
                Tensor
                    updated memory elements, with shape `(M, B, D)`.
                Tensor
                    attention key computed for left context and utterance.
                Tensor
                    attention value computed for left context and utterance.
        """
        query_dim = right_context.size(0) + utterance.size(0) + summary.size(0)
        key_dim = right_context.size(0) + utterance.size(0) + mems.size(0) + left_context_key.size(0)
        attention_mask = torch.zeros(query_dim, key_dim).to(dtype=torch.bool, device=utterance.device)
        attention_mask[-1, :mems.size(0)] = True
        output, output_mems, key, value = self._forward_impl(utterance, lengths, right_context, summary, mems, attention_mask, left_context_key=left_context_key, left_context_val=left_context_val)
        return (output, output_mems, key[mems.size(0) + right_context.size(0):], value[mems.size(0) + right_context.size(0):])