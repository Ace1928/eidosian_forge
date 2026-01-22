import json
import math
import os
import re
from typing import Dict, List, Optional, Set
import torch
import torch.utils.benchmark as benchmark
from torch._C._profiler import (
from torch.profiler import profile
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
class OptimizerSingleTensorPattern(Pattern):
    """
    This pattern identifies if we are using the single-tensor version of an optimizer.
    example:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    By adding foreach=True to enable multi-tensor optimizer, we can gain speedup when
    the kernels are relatively small.

    Pattern:
    XXXXX: _single_tenser_<OPTIMIZER_NAME>

    Algorithm:
    String match
    """

    def __init__(self, prof: profile, should_benchmark: bool=False):
        super().__init__(prof, should_benchmark)
        self.name = 'Optimizer Single Tensor Pattern'
        self.optimizers_with_foreach = ['adam', 'sgd', 'adamw']
        self.description = "Deteced optimizer running with single tensor implementation. Please enable multi tensor implementation by passing 'foreach=True' into optimizer."
        self.url = ''

    def match(self, event: _ProfilerEvent):
        for optimizer in self.optimizers_with_foreach:
            if event.name.endswith(f'_single_tensor_{optimizer}'):
                return True
        return False