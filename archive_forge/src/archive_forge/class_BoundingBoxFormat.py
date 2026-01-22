from __future__ import annotations
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import torch
from torch.utils._pytree import tree_flatten
from ._tv_tensor import TVTensor
class BoundingBoxFormat(Enum):
    """[BETA] Coordinate format of a bounding box.

    Available formats are

    * ``XYXY``
    * ``XYWH``
    * ``CXCYWH``
    """
    XYXY = 'XYXY'
    XYWH = 'XYWH'
    CXCYWH = 'CXCYWH'