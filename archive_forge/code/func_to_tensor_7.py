from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def to_tensor_7(self) -> torch.Tensor:
    """
        Converts a transformation to a tensor with 7 final columns, four for the quaternion followed by three for the
        translation.

        Returns:
            A [*, 7] tensor representation of the transformation
        """
    tensor = self._trans.new_zeros((*self.shape, 7))
    tensor[..., :4] = self._rots.get_quats()
    tensor[..., 4:] = self._trans
    return tensor