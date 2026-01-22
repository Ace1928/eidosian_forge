import math
from typing import List, Optional
import torch
from torch import nn, Tensor
from .image_list import ImageList

    This module generates the default boxes of SSD for a set of feature maps and image sizes.

    Args:
        aspect_ratios (List[List[int]]): A list with all the aspect ratios used in each feature map.
        min_ratio (float): The minimum scale :math:`	ext{s}_{	ext{min}}` of the default boxes used in the estimation
            of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
        max_ratio (float): The maximum scale :math:`	ext{s}_{	ext{max}}`  of the default boxes used in the estimation
            of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
        scales (List[float]], optional): The scales of the default boxes. If not provided it will be estimated using
            the ``min_ratio`` and ``max_ratio`` parameters.
        steps (List[int]], optional): It's a hyper-parameter that affects the tiling of default boxes. If not provided
            it will be estimated from the data.
        clip (bool): Whether the standardized values of default boxes should be clipped between 0 and 1. The clipping
            is applied while the boxes are encoded in format ``(cx, cy, w, h)``.
    