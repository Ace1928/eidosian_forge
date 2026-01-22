from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
def _check_scale_growth_tracker(self, funcname: str) -> Tuple[torch.Tensor, torch.Tensor]:
    fix = 'This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration.'
    assert self._scale is not None, f'Attempted {funcname} but _scale is None.  ' + fix
    assert self._growth_tracker is not None, f'Attempted {funcname} but _growth_tracker is None.  ' + fix
    return (self._scale, self._growth_tracker)