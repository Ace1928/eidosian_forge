from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter
from ..kvstore import KVStore
def _allreduce_grads(self):
    if not self._kvstore:
        return
    for i, param in enumerate(self._params):
        if param.grad_req != 'null':
            grad_list = param.list_grad()
            if grad_list[0].stype != 'default':
                self._kvstore.push(i, grad_list, priority=-i)
                if param._stype == 'default':
                    if self._update_on_kvstore:
                        pull_list = param.list_data()
                    else:
                        pull_list = param.list_grad()
                    self._kvstore.pull(i, pull_list, priority=-i, ignore_sparse=self._distributed)
            elif self._update_on_kvstore:
                self._kvstore.pushpull(i, grad_list, out=param.list_data(), priority=-i)
            else:
                self._kvstore.pushpull(i, grad_list, priority=-i)