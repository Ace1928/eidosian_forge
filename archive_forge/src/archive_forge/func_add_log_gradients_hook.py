import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def add_log_gradients_hook(self, module: 'torch.nn.Module', name: str='', prefix: str='', log_freq: int=0) -> None:
    """This instruments hooks into the pytorch module
        log gradients after a backward pass
        log_freq - log gradients/parameters every N batches
        """
    prefix = prefix + name
    if not hasattr(module, '_wandb_hook_names'):
        module._wandb_hook_names = []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            log_track_grad = log_track_init(log_freq)
            module._wandb_hook_names.append('gradients/' + prefix + name)
            self._hook_variable_gradient_stats(parameter, 'gradients/' + prefix + name, log_track_grad)