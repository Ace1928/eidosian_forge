from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter
from ..kvstore import KVStore
def allreduce_grads(self):
    """For each parameter, reduce the gradients from different contexts.

        Should be called after `autograd.backward()`, outside of `record()` scope,
        and before `trainer.update()`.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        """
    if not self._kv_initialized:
        self._init_kvstore()
    if self._params_to_init:
        self._init_params()
    assert not (self._kvstore and self._update_on_kvstore), 'allreduce_grads() when parameters are updated on kvstore is not supported. Try setting `update_on_kvstore` to False when creating trainer.'
    self._allreduce_grads()