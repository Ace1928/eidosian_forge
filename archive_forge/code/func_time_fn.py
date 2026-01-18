import sys
import torch.utils.benchmark as benchmark_utils
def time_fn(m):
    return m.median / m.metadata['numel']