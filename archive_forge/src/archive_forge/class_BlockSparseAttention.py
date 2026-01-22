import logging
import math
from dataclasses import dataclass
import torch
from xformers import _is_triton_available
from xformers.components.attention import Attention, AttentionConfig, register_attention
@register_attention('blocksparse', BlockSparseAttentionConfig)
class BlockSparseAttention(Attention):
    """
        Thin wrap over the Triton blocksparse computations. The sparsity pattern is determined through the layout.

        .. warning: the layout is assumed to have the dimensions [heads, seq, seq].
            If some dimensions are missing, we assume that the same layout is to be used across heads.

        .. warning: for now, the sequence (context) length has to be a power of two. This constraint could
            be relaxed in the future.

        .. warning: the block size has to be picked from [16, 32, 64]. Some speed is gained from bigger blocks.
            It is of course possible to reproduce coarser patterns given these primitives, as the user sees fit.

        """

    def __init__(self, layout: torch.Tensor, block_size: int=16, dropout: float=0.0, num_heads: int=1, causal: bool=False, *args, **kwargs):
        if layout.dim() == 2:
            logger.warning('The layout passed is lacking a head dimension and a batch dimension')
            logger.warning('Now assuming that the same layout is to be used across all heads')
            layout = layout.unsqueeze(0).expand(num_heads, -1, -1)
            logger.warning(f'New layout dimensions: {layout.shape}')
        assert block_size in (16, 32, 64, 128), 'Only block sizes in [16, 32, 64, 128] are supported'
        super().__init__()
        self.causal = causal
        self.attn_drop = torch.nn.Dropout(dropout, inplace=False)
        self.layout = layout
        self.block_size = block_size
        self.requires_head_dimension = True
        self.requires_same_k_q_dimensions = True
        self.supports_attention_mask = False
        self.supports_key_padding_mask = False

    def create_triton_kernels(self, device):
        self.sparse_dot_sdd = blocksparse_matmul(self.layout, self.block_size, 'sdd', trans_a=False, trans_b=True, device=device)
        self.sparse_dot_dsd = blocksparse_matmul(self.layout, self.block_size, 'dsd', trans_a=False, trans_b=False, device=device)
        self.sparse_softmax = blocksparse_softmax(self.layout, self.block_size, device=device)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float=1.0, *args, **kwargs) -> torch.Tensor:
        assert 'att_mask' not in kwargs.keys() and 'att_mask' not in args, 'This attention does not support an attention mask, but you can specify causality.'
        '\n            A thin wrap around the Triton blockparse attention operation\n\n            .. note: Per element attention mask is not supported, but you can specify causality\n            '
        if not hasattr(self, 'sparse_dot_sdd'):
            self.create_triton_kernels(q.device)
        assert q.shape[-2] == k.shape[-2], 'Blocksparse requires the same dimensions for K and Q for now'
        assert q.shape[-2] == self.layout.shape[-2] * self.block_size, 'Actual sequence size and layout are inconsistent'
        assert k.shape[-2] == self.layout.shape[-2] * self.block_size, 'Actual sequence size and layout are inconsistent'
        assert q.shape[-2] % self.block_size == 0, 'Sequence length {}  must be a multiple of block size {}'.format(q.shape[-2], self.block_size)
        q = q / math.sqrt(q.size(-1))
        sparse_att_mat = self.sparse_dot_sdd(q, k)
        sparse_att_mat = self.sparse_softmax(sparse_att_mat, scale=scale, is_causal=self.causal)
        sparse_att_mat = self.attn_drop(sparse_att_mat)
        a = self.sparse_dot_dsd(sparse_att_mat, v)
        return a