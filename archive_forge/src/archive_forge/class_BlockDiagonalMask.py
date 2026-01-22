import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class BlockDiagonalMask(AttentionBias):
    """
    A block-diagonal mask that can be passed as ``attn_bias``
    argument to :attr:`xformers.ops.memory_efficient_attention`.

    Queries and Keys are each divided into the same number of blocks.
    Queries in block i only attend to keys in block i.

    .. figure:: /_static/block_diag_bias.png

        This bias can be used to handle a batch of sequences of
        different lengths, via :attr:`BlockDiagonalMask.from_tensor_list`

    :Example:

    .. code-block:: python

        import torch
        from xformers.ops import fmha

        K = 16
        dtype = torch.float16
        device = "cuda"
        list_x = [
            torch.randn([1, 3, 1, K], dtype=dtype, device=device),
            torch.randn([1, 6, 1, K], dtype=dtype, device=device),
            torch.randn([1, 2, 1, K], dtype=dtype, device=device),
        ]
        attn_bias, x = fmha.BlockDiagonalMask.from_tensor_list(list_x)
        linear = torch.nn.Linear(K, K * 3).to(device=device, dtype=dtype)

        q, k, v = linear(x).reshape([1, -1, 1, 3, K]).unbind(-2)
        out = fmha.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        list_out = attn_bias.split(out)
        print(list_out[0].shape)  # [1, 3, 1, K]
        assert tuple(list_out[0].shape) == (1, 3, 1, K)

    """
    q_seqinfo: _SeqLenInfo
    k_seqinfo: _SeqLenInfo
    _batch_sizes: Optional[Sequence[int]] = None

    def _create_block_mask(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        assert shape[-1] == self.k_seqinfo.seqstart_py[-1], (shape[-1], self.k_seqinfo.seqstart_py[-1])
        assert shape[-2] == self.q_seqinfo.seqstart_py[-1], (shape[-2], self.q_seqinfo.seqstart_py[-1])
        mask = torch.empty(shape[-2:], dtype=dtype, device=device)
        mask.fill_(-math.inf)
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(self.q_seqinfo.intervals(), self.k_seqinfo.intervals())):
            mask[q_start:q_end, k_start:k_end] = self._create_block_mask((q_end - q_start, k_end - k_start), dtype=dtype, device=device)
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.expand(shape)

    @classmethod
    def from_seqlens(cls, q_seqlen: Sequence[int], kv_seqlen: Optional[Sequence[int]]=None) -> 'BlockDiagonalMask':
        """Creates a :attr:`BlockDiagonalMask` from a list of tensors lengths for query and key/value.

        Args:
            q_seqlen (Union[Sequence[int], torch.Tensor]): List or tensor of sequence lengths for query tensors
            kv_seqlen (Union[Sequence[int], torch.Tensor], optional): List or tensor of sequence lengths for key/value.
                    (Defaults to ``q_seqlen``.)
        Returns:
            BlockDiagonalMask
        """
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen)
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
        if kv_seqlen is None or q_seqlen == kv_seqlen:
            k_seqinfo = q_seqinfo
        else:
            k_seqinfo = _SeqLenInfo.from_seqlens(kv_seqlen)
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)

    @classmethod
    def from_tensor_list(cls, tensors: Sequence[torch.Tensor]) -> Tuple['BlockDiagonalMask', torch.Tensor]:
        """Creates a :attr:`BlockDiagonalMask` from a list of tensors, and returns the tensors
        concatenated on the sequence length dimension

        .. figure:: /_static/block_diag_cat_split.png

            See also :attr:`BlockDiagonalMask.split` to split the returned
            :attr:`torch.Tensor` back to a list of tensors of varying sequence length

        Args:
            tensors (Sequence[torch.Tensor]): A list of tensors of shape ``[B, M_i, *]``.
                All tensors should have the same dimension and the same batch size ``B``, but
                they can have different sequence length ``M``.

        Returns:
            Tuple[BlockDiagonalMask, torch.Tensor]: The corresponding bias for the attention
            along with `tensors` concatenated on the sequence length dimension, with shape ``[1, sum_i{M_i}, *]``
        """
        batch_sizes = [tensor.shape[0] for tensor in tensors]
        seqlens = []
        for x in tensors:
            for _ in range(x.shape[0]):
                seqlens.append(x.shape[1])
        block_diag = cls.from_seqlens(seqlens)
        block_diag._batch_sizes = batch_sizes
        tensors_bs1 = tuple((x.reshape([1, -1, *x.shape[2:]]) for x in tensors))
        concat_tensors = torch.cat(tensors_bs1, dim=1)
        return (block_diag, concat_tensors)

    @classmethod
    def from_tensor_lists_qkv(cls, tensors_q: Sequence[torch.Tensor], tensors_k: Sequence[torch.Tensor], tensors_v: Optional[Sequence[torch.Tensor]]=None) -> Tuple['BlockDiagonalMask', torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert len(tensors_q) == len(tensors_k)
        assert tensors_v is None or len(tensors_v) == len(tensors_q)
        batch_sizes = [tensor.shape[0] for tensor in tensors_q]
        q_seqlens, kv_seqlens = ([], [])
        for i, (q, k) in enumerate(zip(tensors_q, tensors_k)):
            assert q.shape[0] == k.shape[0]
            q_seqlens += [q.shape[1]] * q.shape[0]
            kv_seqlens += [k.shape[1]] * k.shape[0]
            assert tensors_v is None or tensors_v[i].shape[:2] == k.shape[:2]
        block_diag = cls.from_seqlens(q_seqlens, kv_seqlens)
        block_diag._batch_sizes = batch_sizes
        return (block_diag, torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_q], dim=1), torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_k], dim=1), torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_v], dim=1) if tensors_v is not None else None)

    def split_queries(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.q_seqinfo.split(tensor, self._batch_sizes)

    def split_kv(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.k_seqinfo.split(tensor, self._batch_sizes)

    def split(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        """The inverse operation of :attr:`BlockDiagonalCausalMask.from_tensor_list`

        Args:
            tensor (torch.Tensor): Tensor of tokens of shape ``[1, sum_i{M_i}, *]``

        Returns:
            Sequence[torch.Tensor]: A list of tokens with possibly different sequence lengths
        """
        assert self.q_seqinfo is self.k_seqinfo
        return self.q_seqinfo.split(tensor, self._batch_sizes)

    def make_causal(self) -> 'BlockDiagonalCausalMask':
        """Makes each block causal"""
        return BlockDiagonalCausalMask(q_seqinfo=self.q_seqinfo, k_seqinfo=self.k_seqinfo, _batch_sizes=self._batch_sizes)

    def make_causal_from_bottomright(self) -> 'BlockDiagonalCausalFromBottomRightMask':
        """Makes each block causal with a possible non-causal prefix"""
        return BlockDiagonalCausalFromBottomRightMask(q_seqinfo=self.q_seqinfo, k_seqinfo=self.k_seqinfo, _batch_sizes=self._batch_sizes)

    def make_local_attention(self, window_size: int) -> 'BlockDiagonalCausalLocalAttentionMask':
        """Experimental: Makes each block causal with local attention"""
        return BlockDiagonalCausalLocalAttentionMask(q_seqinfo=self.q_seqinfo, k_seqinfo=self.k_seqinfo, _batch_sizes=self._batch_sizes, _window_size=window_size)

    def make_local_attention_from_bottomright(self, window_size: int) -> 'BlockDiagonalCausalLocalAttentionFromBottomRightMask':
        """Experimental: Makes each block causal with local attention, start from bottom right"""
        return BlockDiagonalCausalLocalAttentionFromBottomRightMask(q_seqinfo=self.q_seqinfo, k_seqinfo=self.k_seqinfo, _batch_sizes=self._batch_sizes, _window_size=window_size)