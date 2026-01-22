import math
import random
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def _mask(self, mask, max_mask_patches):
    delta = 0
    for _attempt in range(10):
        target_area = random.uniform(self.mask_group_min_patches, max_mask_patches)
        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
        height = int(round(math.sqrt(target_area * aspect_ratio)))
        width = int(round(math.sqrt(target_area / aspect_ratio)))
        if width < self.width and height < self.height:
            top = random.randint(0, self.height - height)
            left = random.randint(0, self.width - width)
            num_masked = mask[top:top + height, left:left + width].sum()
            if 0 < height * width - num_masked <= max_mask_patches:
                for i in range(top, top + height):
                    for j in range(left, left + width):
                        if mask[i, j] == 0:
                            mask[i, j] = 1
                            delta += 1
            if delta > 0:
                break
    return delta