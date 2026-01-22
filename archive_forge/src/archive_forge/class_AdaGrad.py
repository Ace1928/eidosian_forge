import logging
import math
import pickle
import warnings
import os
import numpy
from ..base import py_str
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply,
from ..ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
from ..ndarray.contrib import (multi_lamb_update, multi_mp_lamb_update)
from ..ndarray import sparse
from ..random import normal
from ..util import is_np_array
@register
class AdaGrad(Optimizer):
    """AdaGrad optimizer.

    This class implements the AdaGrad optimizer described in *Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization*, and available at
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad, clip_gradient)
        history += square(grad)
        div = grad / sqrt(history + float_stable_eps)
        weight += (div + weight * wd) * -lr

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    See Also
    ----------
    :meth:`mxnet.ndarray.sparse.adagrad_update`.

    Parameters
    ----------
    eps: float, optional
        Initial value of the history accumulator. Avoids division by 0.

    """

    def __init__(self, eps=1e-07, **kwargs):
        super(AdaGrad, self).__init__(**kwargs)
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        return zeros(weight.shape, weight.context, stype=weight.stype)

    def update(self, index, weight, grad, state):
        assert isinstance(weight, NDArray)
        assert isinstance(grad, NDArray)
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        is_sparse = grad.stype == 'row_sparse'
        history = state
        if is_sparse:
            kwargs = {'epsilon': self.float_stable_eps, 'rescale_grad': self.rescale_grad}
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            sparse.adagrad_update(weight, grad, history, out=weight, lr=lr, wd=wd, **kwargs)
        else:
            grad = grad * self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            history[:] += square(grad)
            div = grad / sqrt(history + self.float_stable_eps)
            weight[:] += (div + weight * wd) * -lr