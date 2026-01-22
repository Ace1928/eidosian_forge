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
class LBSGD(Optimizer):
    """The Large Batch SGD optimizer with momentum and weight decay.

    The optimizer updates the weight by::

        state = momentum * state + lr * rescale_grad * clip(grad, clip_gradient) + wd * weight
        weight = weight - state

    For details of the update algorithm see :class:`~mxnet.ndarray.sgd_update`
    and :class:`~mxnet.ndarray.sgd_mom_update`.
    In addition to the SGD updates the LBSGD optimizer uses the LARS, Layer-wise
    Adaptive Rate Scaling, algorithm to have a separate learning rate for each
    layer of the network, which leads to better stability over large batch sizes.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
        The momentum value.
    multi_precision: bool, optional
        Flag to control the internal precision of the optimizer.
        False: results in using the same precision as the weights (default),
        True: makes internal 32-bit copy of the weights and applies gradients
        in 32-bit precision even if actual weights used in the model have lower precision.
        Turning this on can improve convergence and accuracy when training with float16.

    warmup_strategy: string ('linear', 'power2', 'sqrt'. , 'lars'   default : 'linear')
    warmup_epochs: unsigned, default: 5
    batch_scale:   unsigned, default: 1 (same as batch size * numworkers)
    updates_per_epoch: updates_per_epoch (default: 32, Default might not reflect true number batches per epoch. Used for warmup.)
    begin_epoch: unsigned, default 0, starting epoch.
    """

    def __init__(self, momentum=0.0, multi_precision=False, warmup_strategy='linear', warmup_epochs=5, batch_scale=1, updates_per_epoch=32, begin_epoch=0, num_epochs=60, **kwargs):
        super(LBSGD, self).__init__(**kwargs)
        logging.info('Running Large-Batch SGD Algorithm')
        logging.info('(Batch_scale=%f, warmup_epochs=%d, warmup_strategy=%s, updates_per_epoch=%d)', batch_scale, warmup_epochs, warmup_strategy, updates_per_epoch)
        self.momentum = momentum
        self.multi_precision = multi_precision
        self.warmup_strategy = warmup_strategy
        self.warmup_epochs = warmup_epochs
        self.batch_scale = batch_scale
        self.updates_per_epoch = updates_per_epoch
        self.init_updates = begin_epoch * updates_per_epoch
        self.num_epochs = num_epochs
        self.lbmult = 1
        self.cumgrads = {}
        self.adaptive = False
        self.admult = 1

    def create_state(self, index, weight):
        momentum = None
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = array(weight, ctx=weight.context, dtype=numpy.float32)
            if self.momentum != 0.0:
                momentum = zeros(weight.shape, weight.context, dtype=numpy.float32, stype=weight.stype)
            return (momentum, weight_master_copy)
        if weight.dtype == numpy.float16 and (not self.multi_precision):
            warnings.warn('Accumulating with float16 in optimizer can lead to poor accuracy or slow convergence. Consider using multi_precision=True option of the SGD optimizer')
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=weight.stype)
        return momentum

    def _get_lbmult(self, nup):
        """Returns lr scaling factor for large batch according to warmup schedule
        (to be implemented)
        """
        nwup = self.warmup_epochs * self.updates_per_epoch
        strategy = self.warmup_strategy
        maxmult = float(self.batch_scale)
        if nup >= nwup:
            mult = maxmult
        elif nwup <= 1:
            mult = 1.0
        elif strategy == 'linear':
            mult = 1.0 + (maxmult - 1) * nup / nwup
        elif strategy == 'power2':
            mult = 1.0 + (maxmult - 1) * (nup * nup) / (nwup * nwup)
        elif strategy == 'sqrt':
            mult = 1.0 + (maxmult - 1) * math.sqrt(float(nup) / nwup)
        else:
            mult = 1.0
        return mult

    def _get_lars(self, weight, g, wd):
        """Returns a scaling factor for the learning rate for this layer
        default is 1
        """
        weight2 = self._l2norm(weight)
        grad2 = self._l2norm(g)
        lars = math.sqrt(weight2 / (grad2 + wd * weight2 + 1e-18))
        if lars < 0.01:
            lars = 0.01
        elif lars > 100:
            lars = 100
        return lars

    def _l2norm(self, v):
        """inner product implementation"""
        norm = multiply(v, v).asnumpy().sum()
        return norm

    def _reset_cum_gradient(self, index):
        """called every macro-batch to reset cumulated gradients to 0 for a given index"""
        self.cumgrads[index]['cum_grad'] = 0

    def _get_cum_gradient(self, index):
        """get the cumulated gradient for index"""
        if index in self.cumgrads:
            return self.cumgrads[index]
        else:
            return {}

    def _put_cum_gradient(self, index, cgrad):
        """store cumulated gradient for index"""
        self.cumgrads[index] = cgrad

    def _cumulate_gradient(self, grad, index):
        """Cumulate gradients for large-batch emulation. Cumulated by index (layer)"""
        cgrad = self._get_cum_gradient(index)
        if cgrad:
            num_cums = cgrad['num_cums']
            if num_cums > 0:
                cum_grad = cgrad['cum_grad'] + grad
                num_cums += 1
            else:
                cum_grad = grad
                num_cums = self.init_updates + 1
        else:
            cum_grad = grad
            num_cums = self.init_updates + 1
        cgrad = {'cum_grad': cum_grad, 'num_cums': num_cums}
        self._put_cum_gradient(index, cgrad)
        return cgrad

    def update(self, index, weight, grad, state):
        assert isinstance(weight, NDArray)
        assert isinstance(grad, NDArray)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        cgrad = self._cumulate_gradient(grad, index)
        if cgrad['num_cums'] % self.batch_scale == 0:
            grad = cgrad['cum_grad'] / self.batch_scale
            if self.warmup_strategy == 'lars':
                lbmult = self._get_lars(weight, grad, wd)
            else:
                lbmult = self._get_lbmult(cgrad['num_cums'])
            lr = lr * lbmult
            kwargs = {'rescale_grad': self.rescale_grad}
            if self.momentum > 0:
                kwargs['momentum'] = self.momentum
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            use_multi_precision = isinstance(state, (list, tuple))
            if not use_multi_precision:
                if state is not None:
                    sgd_mom_update(weight, grad, state, out=weight, lr=lr, wd=wd, **kwargs)
                else:
                    sgd_update(weight, grad, out=weight, lr=lr, wd=wd, **kwargs)
            elif state[0] is not None:
                mp_sgd_mom_update(weight, grad, state[0], state[1], out=weight, lr=lr, wd=wd, **kwargs)
            else:
                mp_sgd_update(weight, grad, state[1], out=weight, lr=lr, wd=wd, **kwargs)
            self._reset_cum_gradient(index)
        else:
            lr = 0.0
            kwargs = {}
            sgd_update(weight, grad, out=weight, lr=lr, wd=wd, **kwargs)