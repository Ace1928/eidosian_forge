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
@pytest.mark.parametrize('data,n_args,kwargs_keys', [(numpy.zeros((2, 3), dtype='f'), 1, []), ([numpy.zeros((2, 3), dtype='f'), numpy.zeros((2, 3), dtype='f')], 2, []), ((numpy.zeros((2, 3), dtype='f'), numpy.zeros((2, 3), dtype='f')), 2, []), ({'a': numpy.zeros((2, 3), dtype='f'), 'b': numpy.zeros((2, 3), dtype='f')}, 0, ['a', 'b']), (ArgsKwargs((numpy.zeros((2, 3), dtype='f'), numpy.zeros((2, 3), dtype='f')), {'c': numpy.zeros((2, 3), dtype='f')}), 2, ['c'])])
def test_pytorch_convert_inputs(data, n_args, kwargs_keys):
    import torch.nn
    model = PyTorchWrapper(torch.nn.Linear(3, 4))
    convert_inputs = model.attrs['convert_inputs']
    Y, backprop = convert_inputs(model, data, is_train=True)
    check_input_converters(Y, backprop, data, n_args, kwargs_keys, torch.Tensor)