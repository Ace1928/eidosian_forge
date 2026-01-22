from collections import defaultdict
from itertools import chain
import pickle
from typing import (
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode
def _plot_figure(x, y_values, labels):
    min_val = min(list(chain(*y_values))) * 0.999
    max_val = max(list(chain(*y_values))) * 1.001
    plt.figure()
    for y, label in zip(y_values, labels):
        plt.plot(x, y, label=label)
    plt.xlabel('# Operator Calls')
    plt.ylabel('Memory (MB)')
    plt.legend()
    for marker_name, marker in self._markers.items():
        if marker_name == 'fw_bw_boundary':
            plt.plot([marker, marker], [min_val, max_val], 'r', lw=2, label=marker_name)
        else:
            plt.plot([marker, marker], [min_val, max_val], 'k-', lw=2, label=marker_name)