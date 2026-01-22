from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
def _maybe_opt_step(self, optimizer: torch.optim.Optimizer, optimizer_state: Dict[str, Any], *args: Any, **kwargs: Any) -> Optional[float]:
    retval: Optional[float] = None
    if not sum((v.item() for v in optimizer_state['found_inf_per_device'].values())):
        retval = optimizer.step(*args, **kwargs)
    return retval