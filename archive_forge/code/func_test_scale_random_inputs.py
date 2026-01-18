import pytest
from hypothesis import given, settings
from hypothesis.strategies import lists, one_of, tuples
from thinc.api import PyTorchGradScaler
from thinc.compat import has_torch, has_torch_amp, has_torch_cuda_gpu, torch
from thinc.util import is_torch_array
from ..strategies import ndarrays
@pytest.mark.skipif(not has_torch, reason='needs PyTorch')
@pytest.mark.skipif(not has_torch_cuda_gpu, reason='needs a GPU')
@pytest.mark.skipif(not has_torch_amp, reason='requires PyTorch with mixed-precision support')
@given(X=one_of(tensors(), lists(tensors()), tuples(tensors())))
@settings(deadline=None)
def test_scale_random_inputs(X):
    import torch
    device_id = torch.cuda.current_device()
    scaler = PyTorchGradScaler(enabled=True)
    scaler.to_(device_id)
    if is_torch_array(X):
        assert torch.allclose(scaler.scale(X), X * 2.0 ** 16)
    else:
        scaled1 = scaler.scale(X)
        scaled2 = [t * 2.0 ** 16 for t in X]
        for t1, t2 in zip(scaled1, scaled2):
            assert torch.allclose(t1, t2)