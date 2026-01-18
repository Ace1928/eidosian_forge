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
@pytest.mark.parametrize('ops_mixed', XP_OPS_MIXED)
@pytest.mark.parametrize('nN,nI,nO', [(2, 3, 4)])
def test_pytorch_wrapper_thinc_input(ops_mixed, nN, nI, nO):
    import torch.nn
    ops, mixed_precision = ops_mixed
    with use_ops(ops.name):
        ops = get_current_ops()
        pytorch_layer = torch.nn.Linear(nO, nO)
        torch.nn.init.uniform_(pytorch_layer.weight, 9.0, 11.0)
        device = get_torch_default_device()
        model = chain(Relu(), PyTorchWrapper_v2(pytorch_layer.to(device), mixed_precision=mixed_precision, grad_scaler=PyTorchGradScaler(enabled=mixed_precision, init_scale=2.0 ** 16)).initialize())
        if isinstance(ops, CupyOps):
            assert 'pytorch' in context_pools.get()
        sgd = SGD(0.001)
        X = ops.xp.zeros((nN, nI), dtype='f')
        X += ops.xp.random.uniform(size=X.size).reshape(X.shape)
        Y = ops.xp.zeros((nN, nO), dtype='f')
        model.initialize(X, Y)
        Yh, get_dX = model.begin_update(X)
        assert isinstance(Yh, ops.xp.ndarray)
        assert Yh.shape == (nN, nO)
        dYh = (Yh - Y) / Yh.shape[0]
        dX = get_dX(dYh)
        model.finish_update(sgd)
        assert dX.shape == (nN, nI)
        check_learns_zero_output(model, sgd, X, Y)
        assert isinstance(model.predict(X), ops.xp.ndarray)