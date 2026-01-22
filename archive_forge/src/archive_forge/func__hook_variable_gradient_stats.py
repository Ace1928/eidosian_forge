import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def _hook_variable_gradient_stats(self, var, name, log_track):
    """Logs a Variable's gradient's distribution statistics next time backward()
        is called on it.
        """
    if not isinstance(var, torch.autograd.Variable):
        cls = type(var)
        raise TypeError(f'Expected torch.Variable, not {cls.__module__}.{cls.__name__}')
    handle = self._hook_handles.get(name)
    if handle is not None and self._torch_hook_handle_is_valid(handle):
        raise ValueError(f'A hook has already been set under name "{name}"')

    def _callback(grad, log_track):
        if not log_track_update(log_track):
            return
        self.log_tensor_stats(grad.data, name)
    handle = var.register_hook(lambda grad: _callback(grad, log_track))
    self._hook_handles[name] = handle
    return handle