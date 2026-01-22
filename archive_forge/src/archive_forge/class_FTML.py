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
class FTML(Optimizer):
    """The FTML optimizer.

    This class implements the optimizer described in
    *FTML - Follow the Moving Leader in Deep Learning*,
    available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.

    Denote time step by t. The optimizer updates the weight by::

        rescaled_grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
        v = beta2 * v + (1 - beta2) * square(rescaled_grad)
        d_t = (1 - power(beta1, t)) / lr * square_root(v / (1 - power(beta2, t))) + epsilon)
        z = beta1 * z + (1 - beta1) * rescaled_grad - (d_t - beta1 * d_(t-1)) * weight
        weight = - z / d_t

    For details of the update algorithm, see :class:`~mxnet.ndarray.ftml_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional
        0 < beta1 < 1. Generally close to 0.5.
    beta2 : float, optional
        0 < beta2 < 1. Generally close to 1.
    epsilon : float, optional
        Small value to avoid division by 0.
    """

    def __init__(self, beta1=0.6, beta2=0.999, epsilon=1e-08, **kwargs):
        super(FTML, self).__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype), zeros(weight.shape, weight.context, dtype=weight.dtype), zeros(weight.shape, weight.context, dtype=weight.dtype))

    def update(self, index, weight, grad, state):
        assert isinstance(weight, NDArray)
        assert isinstance(grad, NDArray)
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        t = self._index_update_count[index]
        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon, 'rescale_grad': self.rescale_grad, 't': t}
        if self.clip_gradient:
            kwargs['clip_grad'] = self.clip_gradient
        prev_d, prev_v, prev_z = state
        ftml_update(weight, grad, prev_d, prev_v, prev_z, out=weight, lr=lr, wd=wd, **kwargs)