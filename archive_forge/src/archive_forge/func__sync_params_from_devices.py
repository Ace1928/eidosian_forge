import logging
import warnings
from .. import context as ctx
from .. import optimizer as opt
from .. import ndarray as nd
from .executor_group import DataParallelExecutorGroup
from ..model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore
from ..model import load_checkpoint
from ..initializer import Uniform, InitDesc
from ..io import DataDesc
from ..ndarray import zeros
from .base_module import BaseModule, _check_input_names, _parse_data_desc
def _sync_params_from_devices(self):
    """Synchronizes parameters from devices to CPU. This function should be called after
        calling `update` that updates the parameters on the devices, before one can read the
        latest parameters from ``self._arg_params`` and ``self._aux_params``.

        For row_sparse parameters on devices, ther are pulled from KVStore with all row ids.

        """
    self._exec_group.get_params(self._arg_params, self._aux_params)
    if self._kvstore and self._update_on_kvstore:
        for param_name, param_val in sorted(self._arg_params.items()):
            if param_val.stype == 'row_sparse':
                row_ids = nd.arange(0, param_val.shape[0], dtype='int64')
                self._kvstore.row_sparse_pull(param_name, param_val, row_ids=row_ids)
    self._params_dirty = False