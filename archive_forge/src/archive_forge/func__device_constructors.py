import torch
from torch.overrides import TorchFunctionMode
from torch.utils._contextlib import context_decorator
import functools
@functools.lru_cache(1)
def _device_constructors():
    return {torch.empty, torch.empty_permuted, torch.empty_strided, torch.empty_quantized, torch.ones, torch.arange, torch.bartlett_window, torch.blackman_window, torch.eye, torch.fft.fftfreq, torch.fft.rfftfreq, torch.full, torch.fill, torch.hamming_window, torch.hann_window, torch.kaiser_window, torch.linspace, torch.logspace, torch.nested.nested_tensor, torch.ones, torch.rand, torch.randn, torch.randint, torch.randperm, torch.range, torch.sparse_coo_tensor, torch.sparse_compressed_tensor, torch.sparse_csr_tensor, torch.sparse_csc_tensor, torch.sparse_bsr_tensor, torch.sparse_bsc_tensor, torch.tril_indices, torch.triu_indices, torch.vander, torch.zeros, torch.asarray, torch.tensor, torch.as_tensor, torch.scalar_tensor, torch.asarray}