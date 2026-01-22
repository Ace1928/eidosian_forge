from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
class SignalSparsity:
    """
    This class represents a particular config for a set of signal
    processing based sparsification functions on tensors. This can
    be used both on weights, gradients and other tensors like the
    optimizer state.

    During initialization, this class requires a value for one of
    `sst_top_k_element` or `sst_top_k_percent` and also requires a
    value for one of `dst_top_k_element` or `dst_top_k_percent`.

    This class only handles tensor inputs and outputs. We leave
    state_dict type of data handling to upper layer functions.

    Args:
        algo (Algo):
            The algorithm used. Default: FFT
        sst_top_k_dim (int, optional):
            The dimension on which the top-k is done for SST.
            E.g. -1 is the last dim. None means flatten and top-k on all dims.
            There is no way to specify multiple dims other than None.
            Default: -1
        sst_top_k_element (int, optional):
            Number of top-k elements to retain for SST. Default: None
        sst_top_k_percent (float, optional):
            Percent of top-k elements to retain for SST. Default: None
        dst_top_k_dim (int, optional):
            The dimension on which the top-k is done for DST.
            E.g. -1 is the last dim. None means flatten and top-k on all dims.
            There is no way to specify multiple dims other than None.
            Default: None
        dst_top_k_element (int, optional):
            Number of top-k elements to retain for DST. Default: None
        dst_top_k_percent (float, optional):
            Percent of top-k elements to retain for DST. Default: None

    Example:
        .. code-block:: python

            2d_sparser = SignalSparsity(sst_top_k_element=10, dst_top_k_element=1)
            sst = 2d_sparser.dense_to_sst(linear.weight.data)

            3d_sparser = SingalSparsity(algo=Algo.FFT, sst_top_k_dim=None, dst_top_k_dim=-1, sst_top_k_percent=10, dst_top_k_element=100)
            conv.weight.data, _, _ = 3d_sparser.lossy_compress(conv.weight.data)
    """

    def __init__(self, algo: Algo=Algo.FFT, sst_top_k_dim: Optional[int]=-1, sst_top_k_element: Optional[int]=None, sst_top_k_percent: Optional[float]=None, dst_top_k_dim: Optional[int]=-1, dst_top_k_element: Optional[int]=None, dst_top_k_percent: Optional[float]=None) -> None:
        self._sst_top_k_dim = sst_top_k_dim
        self._sst_top_k_element = sst_top_k_element
        self._sst_top_k_percent = sst_top_k_percent
        self._dst_top_k_dim = dst_top_k_dim
        self._dst_top_k_element = dst_top_k_element
        self._dst_top_k_percent = dst_top_k_percent
        self._validate_conf()
        self._transform, self._inverse_transform = (_fft_transform, _ifft_transform) if algo is Algo.FFT else (_dct_transform, _idct_transform)

    @property
    def _sst_enabled(self) -> bool:
        """True if SST is enabled."""
        return self._sst_top_k_element is not None or self._sst_top_k_percent is not None

    @property
    def _dst_enabled(self) -> bool:
        """True if DST is enabled."""
        return self._dst_top_k_element is not None or self._dst_top_k_percent is not None

    def _validate_conf(self) -> None:
        """Validating if the config is valid.

        This includes asserting the following:
        1. validating that one and only one of top_k_element and top_k_percent is set.
        2. Asserting that both element and percentage are in valid ranges.

        Throws:
            ValueError:
                If validation fails.
        """

        def both_set(a: Optional[int], b: Optional[float]) -> bool:
            return a is not None and b is not None
        if both_set(self._sst_top_k_element, self._sst_top_k_percent) or both_set(self._dst_top_k_element, self._dst_top_k_percent):
            raise ValueError(f"top_k_element and top_k_percent can't be both set\nInput values are: sst element={self._sst_top_k_element}, sst percent={self._sst_top_k_percent}, dst element={self._dst_top_k_element}, dst percent={self._dst_top_k_percent}")

        def none_or_in_range(a: Optional[float]) -> bool:
            return a is None or 0.0 < a <= 100.0
        if not (none_or_in_range(self._sst_top_k_percent) and none_or_in_range(self._dst_top_k_percent)):
            raise ValueError(f'top_k_percent values for sst and dst has to be in the interval (0, 100].\nInput values are: sst percent={self._sst_top_k_percent}, dst percent={self._dst_top_k_percent}')

        def none_or_greater_0(a: Optional[int]) -> bool:
            return a is None or 0 < a
        if not (none_or_greater_0(self._sst_top_k_element) and none_or_greater_0(self._dst_top_k_element)):
            raise ValueError(f'top_k_element values for sst and dst has to be greater than 0.\nInput values are: sst element={self._sst_top_k_element} and dst element={self._dst_top_k_element}')

    def dense_to_sst(self, dense: Tensor) -> Optional[Tensor]:
        """Get Signal Sparse Tensor (SST) from a dense tensor

        Dense -> fft -> top-k -> results.

        The input dense tensor is transformed using a transform algorithm according to the `algo`
        initialization argument. The SST is then generated from the top_k_elements
        (or the top_k_percentage) of values from the transformed tensor along the 'sst_top_k_dim'.

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).

        Returns:
            (Tensor, optional):
                Same shaped tensor as the input dense tensor, still in dense format but in frequency
                domain (complex valued) and has zeros.
        """
        if not self._sst_enabled:
            return None
        top_k_total_size = _top_k_total_size(dense, self._sst_top_k_dim)
        k = _get_k_for_topk(self._sst_top_k_percent, self._sst_top_k_element, top_k_total_size)
        dense_freq = self._transform(dense, dim=self._sst_top_k_dim)
        real_dense_freq = dense_freq.real.abs()
        return _scatter_topk_to_sparse_tensor(real_dense_freq, dense_freq, k, dim=self._sst_top_k_dim)

    def dense_sst_to_dst(self, dense: Tensor, sst: Optional[Tensor]) -> Optional[Tensor]:
        """Calculates DST from input dense and SST tensors.

        dense - inverse_transform(sst)[using sst_dst_to_dense method] -> top-k -> dst

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).
            sst (Tensor):
                Input SST tensor (has zeros).

        Returns:
            (Tensor):
                Same shaped tensor, still dense format but has zeros. Non-zeros are top-k delta values.
        """
        if not self._dst_enabled:
            return None
        if sst is None:
            sst = torch.zeros_like(dense, dtype=torch.complex64)
        if not dense.shape == sst.shape:
            raise ValueError('dense and sst have different shapes!')
        top_k_total_size = _top_k_total_size(dense, self._dst_top_k_dim)
        k = _get_k_for_topk(self._dst_top_k_percent, self._dst_top_k_element, top_k_total_size)
        delta = dense - self.sst_dst_to_dense(sst)
        del dense
        return _scatter_topk_to_sparse_tensor(delta.abs(), delta, k, dim=self._dst_top_k_dim)

    def sst_dst_to_dense(self, sst: Optional[Tensor], dst: Optional[Tensor]=None) -> Tensor:
        """From SST and DST returns a dense reconstructed tensor (RT). When argument dst=None, simply returns
        the inverse transform of the SST tensor.

        Args:
            sst (Tensor):
                Singal sparse tensor. Required argument.
            dst (Tensor, optional):
                Delta sparse tensor, optional.

        Returns:
            (Tensor):
                A dense tensor in real number domain from the SST.
        """
        assert not (sst is None and dst is None), 'both-None-case is not useful'
        if sst is None:
            return dst
        dense_rt = torch.real(self._inverse_transform(sst, dim=self._sst_top_k_dim))
        if dst is not None:
            dense_rt += dst
        return dense_rt

    def lossy_compress(self, dense: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """From dense tensor to lossy reconstruction of dense tensor with the help of SST and DST
        tensor calculation. If requested sparsity is zero (or top_100_percent) then simply returns
        the input dense tensor as the reconstruction.

        Args:
            dense (Tensor):
                Input dense tensor (no zeros).

        Returns:
            (Tuple[Tensor, Tensor, Tensor]):
                A tuple of the form (lossy_reconstruction, sst, dst) with three tensors of the same
                shape as the dense tensor.
        """
        if _is_sparsity_zero(dense, self._sst_top_k_percent, self._sst_top_k_element, self._sst_top_k_dim) and _is_sparsity_zero(dense, self._dst_top_k_percent, self._dst_top_k_element, self._dst_top_k_dim):
            return (dense, None, dense)
        else:
            sst = self.dense_to_sst(dense)
            dst = self.dense_sst_to_dst(dense, sst)
            return (self.sst_dst_to_dense(sst, dst), sst, dst)