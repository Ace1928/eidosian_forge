import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def _batched_mask_to_box_tf(masks: 'tf.Tensor'):
    """
    Computes the bounding boxes around the given input masks. The bounding boxes are in the XYXY format which
    corresponds the following required indices:
        - LEFT: left hand side of the bounding box
        - TOP: top of the bounding box
        - RIGHT: right of the bounding box
        - BOTTOM: bottom of the bounding box

    Return [0,0,0,0] for an empty mask. For input shape channel_1 x channel_2 x ... x height x width, the output shape
    is channel_1 x channel_2 x ... x 4.

    Args:
        - masks (`tf.Tensor` of shape `(batch, nb_mask, height, width)`)
    """
    if tf.size(masks) == 0:
        return tf.zeros([*masks.shape[:-2], 4])
    shape = shape_list(masks)
    height, width = shape[-2:]
    in_height = tf.reduce_max(masks, axis=-1)
    in_height_coords = in_height * tf.range(height)[None, :]
    bottom_edges = tf.reduce_max(in_height_coords, axis=-1)
    in_height_coords = in_height_coords + height * ~in_height
    top_edges = tf.reduce_min(in_height_coords, axis=-1)
    in_width, _ = tf.reduce_max(masks, axis=-2)
    in_width_coords = in_width * tf.range(width)[None, :]
    right_edges, _ = tf.reduce_max(in_width_coords, axis=-1)
    in_width_coords = in_width_coords + width * ~in_width
    left_edges, _ = tf.reduce_min(in_width_coords, axis=-1)
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = tf.stack([left_edges, top_edges, right_edges, bottom_edges], axis=-1)
    out = out * tf.expand_dims(~empty_filter, -1)
    out = tf.reshape(out, *shape[:-2], 4)
    return out