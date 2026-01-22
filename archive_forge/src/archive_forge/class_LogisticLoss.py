import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array
class LogisticLoss(Loss):
    """Calculates the logistic loss (for binary losses only):

    .. math::
        L = \\sum_i \\log(1 + \\exp(- {pred}_i \\cdot {label}_i))

    where `pred` is the classifier prediction and `label` is the target tensor
    containing values -1 or 1 (0 or 1 if `label_format` is binary).
    `label` and `pred` can have arbitrary shape as long as they have the same number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    label_format : str, default 'signed'
        Can be either 'signed' or 'binary'. If the label_format is 'signed', all label values should
        be either -1 or 1. If the label_format is 'binary', all label values should be either
        0 or 1.

    Inputs:
        - **pred**: prediction tensor with arbitrary shape.
        - **label**: truth tensor with values -1/1 (label_format is 'signed')
          or 0/1 (label_format is 'binary'). Must have the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, label_format='signed', **kwargs):
        super(LogisticLoss, self).__init__(weight, batch_axis, **kwargs)
        self._label_format = label_format
        if self._label_format not in ['signed', 'binary']:
            raise ValueError('label_format can only be signed or binary, recieved %s.' % label_format)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        if self._label_format == 'signed':
            label = (label + 1.0) / 2.0
        loss = F.relu(pred) - pred * label + F.Activation(-F.abs(pred), act_type='softrelu')
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)