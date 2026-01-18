from typing import Dict, Iterable, List, Union, cast
from ..compat import has_torch_amp, torch
from ..util import is_torch_array
def to_(self, device):
    self._growth_tracker = self._growth_tracker.to(device)
    self._scale = self._scale.to(device)