from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter
from ..kvstore import KVStore
def _row_sparse_pull(self, parameter, out, row_id, full_idx=False):
    """Internal method to invoke pull operations on KVStore. If `full_idx` is set to True,
        `kv.pull` is preferred instead of `kv.row_sparse_pull`.
        """
    if not self._kv_initialized:
        self._init_kvstore()
    if self._params_to_init:
        self._init_params()
    idx = self._param2idx[parameter.name]
    if full_idx and 'dist' not in self._kvstore.type:
        assert row_id.size == out.shape[0]
        self._kvstore.pull(idx, out=out, priority=-idx, ignore_sparse=False)
    else:
        self._kvstore.row_sparse_pull(idx, out=out, row_ids=row_id, priority=-idx)