import warnings
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
from .image_utils import (
from .utils import ExplicitEnum, TensorType, is_jax_tensor, is_tf_tensor, is_torch_tensor
from .utils.import_utils import (
def _center_to_corners_format_torch(bboxes_center: 'torch.Tensor') -> 'torch.Tensor':
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack([center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height], dim=-1)
    return bbox_corners