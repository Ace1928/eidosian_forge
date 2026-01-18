import inspect
import platform
from typing import Tuple, cast
import numpy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers
from numpy.testing import assert_allclose
from packaging.version import Version
from thinc.api import (
from thinc.backends._custom_kernels import KERNELS, KERNELS_LIST, compile_mmh
from thinc.compat import has_cupy_gpu, has_torch, torch_version
from thinc.types import Floats2d
from thinc.util import torch2xp, xp2torch
from .. import strategies
from ..strategies import arrays_BI, ndarrays_of_shape
def torch_softmax_with_temperature(X: Floats2d, dY: Floats2d, temperature: float) -> Tuple[Floats2d, Floats2d]:
    import torch
    Xt = xp2torch(X, requires_grad=True)
    dYt = xp2torch(dY)
    Xt_temp = Xt / temperature
    Yt = torch.nn.functional.softmax(Xt_temp, dim=-1)
    Yt.backward(dYt)
    return (cast(Floats2d, torch2xp(Yt)), cast(Floats2d, torch2xp(cast(torch.Tensor, Xt.grad))))