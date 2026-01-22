import inspect
import warnings
import torch
from .state import AcceleratorState, GradientState
from .utils import DistributedType, honor_type, is_torch_xla_available
def _switch_parameters(self, parameters_map):
    for param_group in self.optimizer.param_groups:
        param_group['params'] = [parameters_map.get(p, p) for p in param_group['params']]