import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array
class L2Loss(Loss):
    """Calculates the mean squared error between `label` and `pred`.

    .. math:: L = \\frac{1}{2} \\sum_i \\vert {label}_i - {pred}_i \\vert^2.

    `label` and `pred` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=1.0, batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.np.square(label - pred) if is_np_array() else F.square(label - pred)
        loss = _apply_weighting(F, loss, self._weight / 2, sample_weight)
        if is_np_array():
            if F is ndarray:
                return F.np.mean(loss, axis=tuple(range(1, loss.ndim)))
            else:
                return F.npx.batch_flatten(loss).mean(axis=1)
        else:
            return F.mean(loss, axis=self._batch_axis, exclude=True)