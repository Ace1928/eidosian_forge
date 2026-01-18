import numpy
import pytest
from thinc.api import (
from thinc.backends import context_pools
from thinc.compat import has_cupy_gpu, has_torch, has_torch_amp, has_torch_mps_gpu
from thinc.layers.pytorchwrapper import PyTorchWrapper_v3
from thinc.shims.pytorch import (
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.util import get_torch_default_device
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_torch, reason='needs PyTorch')
def test_pytorch_roundtrip_conversion():
    import torch
    xp_tensor = numpy.zeros((2, 3), dtype='f')
    torch_tensor = xp2torch(xp_tensor)
    assert isinstance(torch_tensor, torch.Tensor)
    new_xp_tensor = torch2xp(torch_tensor)
    assert numpy.array_equal(xp_tensor, new_xp_tensor)