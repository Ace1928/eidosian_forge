import numpy as np
from typing import Union, Tuple, Any, List
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
@DeveloperAPI
def valid_padding(in_size: Tuple[int, int], filter_size: Union[int, Tuple[int, int]], stride_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Emulates TF Conv2DLayer "valid" padding (no padding) and computes output dims.

    This method, analogous to its "same" counterpart, but it only computes the output
    image size, since valid padding means (0, 0, 0, 0).

    See www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

    Args:
        in_size: Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size: Rows (Height), column (Width) for filter

    Returns:
        The output shape after padding and convolution.
    """
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = (filter_size, filter_size)
    else:
        filter_height, filter_width = filter_size
    if isinstance(stride_size, (int, float)):
        stride_height, stride_width = (int(stride_size), int(stride_size))
    else:
        stride_height, stride_width = (int(stride_size[0]), int(stride_size[1]))
    out_height = int(np.ceil((in_height - filter_height + 1) / float(stride_height)))
    out_width = int(np.ceil((in_width - filter_width + 1) / float(stride_width)))
    return (out_height, out_width)