from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
def _get_growth_tracker(self) -> int:
    if self._enabled:
        return self._init_growth_tracker if self._growth_tracker is None else cast(int, self._growth_tracker.item())
    return 0