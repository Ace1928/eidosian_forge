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
def test_pytorch_wrapper_custom_serde():
    import torch.nn

    def serialize(model):
        return default_serialize_torch_model(model)

    def deserialize(model, state_bytes, device):
        return default_deserialize_torch_model(model, state_bytes, device)

    def get_model():
        return PyTorchWrapper_v3(torch.nn.Linear(2, 3), serialize_model=serialize, deserialize_model=deserialize)
    model = get_model()
    model_bytes = model.to_bytes()
    get_model().from_bytes(model_bytes)
    with make_tempdir() as path:
        model_path = path / 'model'
        model.to_disk(model_path)
        new_model = get_model().from_bytes(model_bytes)
        new_model.from_disk(model_path)