from collections import defaultdict
from itertools import chain
import pickle
from typing import (
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode
def _add_marker(self, marker_name: str) -> None:
    """Set the marker's x-axis value."""
    marker_val = len(self.memories_allocated.values())
    self._markers[marker_name] = marker_val