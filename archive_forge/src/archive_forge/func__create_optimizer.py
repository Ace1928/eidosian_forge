import time
import logging
import mxnet as mx
from mxnet.module import Module
from .svrg_optimizer import _SVRGOptimizer
def _create_optimizer(self, optimizer, default_opt, kvstore, optimizer_params):
    """Helper function to create a svrg optimizer. SVRG optimizer encapsulates two optimizers and
        will redirect update() to the correct optimizer based on the key.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer: str
            Name for SVRGOptimizer
        default_opt : str or Optimizer that was passed in.
        optimizer_params : dict
           optimizer params that was passed in.
        """
    batch_size = self._exec_group.batch_size
    kv_store, update_on_kvstore = mx.model._create_kvstore(kvstore, self._ctx_len, self._arg_params)
    if kv_store and 'dist' in kv_store.type and ('_sync' in kv_store.type):
        batch_size *= kv_store.num_workers
    rescale_grad = 1.0 / batch_size
    idx2name = {}
    if update_on_kvstore:
        idx2name.update(enumerate(self._exec_group.param_names))
    else:
        for k in range(self._ctx_len):
            idx2name.update({i * self._ctx_len + k: n for i, n in enumerate(self._exec_group.param_names)})
    for key in self._param_dict[0].keys():
        max_key = max(list(idx2name.keys())) + 1
        idx2name[max_key] = key + '_full'
    optimizer_params = dict(optimizer_params)
    if 'rescale_grad' not in optimizer_params:
        optimizer_params['rescale_grad'] = rescale_grad
    optimizer_params['default_optimizer'] = default_opt
    optimizer_params['param_idx2name'] = idx2name
    optimizer = mx.optimizer.create(optimizer, **optimizer_params)
    return optimizer