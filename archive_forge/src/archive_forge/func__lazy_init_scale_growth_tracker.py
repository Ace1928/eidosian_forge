from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
def _lazy_init_scale_growth_tracker(self, dev: torch.device) -> None:
    assert self._growth_tracker is None, '_growth_tracker initialized before _scale'
    self._scale = torch.full((), self._init_scale, dtype=torch.float32, device=dev)
    self._growth_tracker = torch.full((), self._init_growth_tracker, dtype=torch.int32, device=dev)