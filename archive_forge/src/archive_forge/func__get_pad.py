import math
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def _get_pad(size, size_divisor):
    new_size = math.ceil(size / size_divisor) * size_divisor
    pad_size = new_size - size
    pad_size_left = pad_size // 2
    pad_size_right = pad_size - pad_size_left
    return (pad_size_left, pad_size_right)