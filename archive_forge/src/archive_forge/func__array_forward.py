from typing import (
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array2d, Ragged, XY_XY_OutT
from ..util import get_width
from .noop import noop
def _array_forward(model: Model[InT, OutT], X, Ys: List, callbacks, is_train: bool) -> Tuple[Array2d, Callable]:
    widths = [Y.shape[1] for Y in Ys]
    output = model.ops.xp.hstack(Ys)

    def backprop(d_output: Array2d) -> InT:
        dY = model.ops.as_contig(d_output[:, :widths[0]])
        dX = callbacks[0](dY)
        start = widths[0]
        add_gradients = hasattr(dX, '__add__') or hasattr(dX, '__iadd__')
        add_gradients_data = hasattr(dX, 'data') and (hasattr(dX.data, '__add__') or hasattr(dX.data, '__iadd__'))
        for bwd, width in zip(callbacks[1:], widths[1:]):
            dY = model.ops.as_contig(d_output[:, start:start + width])
            gradient = bwd(dY)
            if add_gradients:
                dX += gradient
            elif add_gradients_data:
                dX.data += gradient.data
            start += width
        return dX
    return (output, backprop)