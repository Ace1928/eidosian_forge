import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def _get_preprocess_shape(self, old_shape: Tuple[int, int], longest_edge: int):
    """
        Compute the output size given input size and target long side length.
        """
    oldh, oldw = old_shape
    scale = longest_edge * 1.0 / max(oldh, oldw)
    newh, neww = (oldh * scale, oldw * scale)
    newh = int(newh + 0.5)
    neww = int(neww + 0.5)
    return (newh, neww)