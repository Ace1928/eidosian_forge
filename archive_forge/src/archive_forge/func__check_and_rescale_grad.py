from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter
from ..kvstore import KVStore
def _check_and_rescale_grad(self, scale):
    if self._update_on_kvstore and self._distributed and self._kv_initialized:
        if self._optimizer.rescale_grad != scale:
            raise UserWarning('Possible change in the `batch_size` from previous `step` detected. Optimizer gradient normalizing factor will not change w.r.t new batch_size when update_on_kvstore=True and when distributed kvstore is used.')
    self._optimizer.rescale_grad = scale