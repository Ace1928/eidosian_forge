from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
def to_gpu(self):
    for gindex, group in enumerate(self.param_groups):
        for pindex, p in enumerate(group['params']):
            if p in self.state:
                values = self.state[p]
                for k, v in values.items():
                    if isinstance(v, torch.Tensor):
                        is_paged = getattr(v, 'is_paged', False)
                        if not is_paged:
                            self.state[p][k] = v.to(p.device)