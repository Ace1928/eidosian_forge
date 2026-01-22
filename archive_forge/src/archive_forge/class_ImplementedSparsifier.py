from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class ImplementedSparsifier(BaseSparsifier):

    def __init__(self, **kwargs):
        super().__init__(defaults=kwargs)

    def update_mask(self, module, **kwargs):
        module.parametrizations.weight[0].mask[0] = 0
        linear_state = self.state['linear1.weight']
        linear_state['step_count'] = linear_state.get('step_count', 0) + 1