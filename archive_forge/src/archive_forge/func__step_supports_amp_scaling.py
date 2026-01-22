from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
import torch
@property
def _step_supports_amp_scaling(self) -> bool:
    return False