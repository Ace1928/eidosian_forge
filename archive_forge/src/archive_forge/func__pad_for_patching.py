import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict, select_best_resolution
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
def _pad_for_patching(self, image: np.array, target_resolution: tuple, input_data_format: ChannelDimension) -> np.array:
    """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
    target_height, target_width = target_resolution
    new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    padded_image = pad(image, padding=((paste_y, paste_y), (paste_x, paste_x)))
    return padded_image