from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
@property
def tile_indices(self):
    if self._tile_indices is None:
        self._tile_indices = get_tile_inds(self.formatB, self.CxB.device)
    return self._tile_indices