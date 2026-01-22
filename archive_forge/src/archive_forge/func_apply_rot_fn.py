from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def apply_rot_fn(self, fn: Callable[[Rotation], Rotation]) -> Rigid:
    """
        Applies a Rotation -> Rotation function to the stored rotation object.

        Args:
            fn: A function of type Rotation -> Rotation
        Returns:
            A transformation object with a transformed rotation.
        """
    return Rigid(fn(self._rots), self._trans)