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
class SGLD(Optimizer):
    """Stochastic Gradient Riemannian Langevin Dynamics.

    This class implements the optimizer described in the paper *Stochastic Gradient
    Riemannian Langevin Dynamics on the Probability Simplex*, available at
    https://papers.nips.cc/paper/4883-stochastic-gradient-riemannian-langevin-dynamics-on-the-probability-simplex.pdf.

    """

    def __init__(self, **kwargs):
        super(SGLD, self).__init__(**kwargs)

    def create_state(self, index, weight):
        return None

    def update(self, index, weight, grad, state):
        assert isinstance(weight, NDArray)
        assert isinstance(grad, NDArray)
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        weight[:] += -lr / 2 * (grad + wd * weight)
        weight[:] += normal(0, math.sqrt(lr), shape=weight.shape, dtype=weight.dtype, ctx=weight.context)