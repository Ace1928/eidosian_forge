import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def _compute_stability_score_tf(masks: 'tf.Tensor', mask_threshold: float, stability_score_offset: int):
    intersections = tf.count_nonzero(masks > mask_threshold + stability_score_offset, axis=[-1, -2], dtype=tf.float32)
    unions = tf.count_nonzero(masks > mask_threshold - stability_score_offset, axis=[-1, -2], dtype=tf.float32)
    stability_scores = intersections / unions
    return stability_scores