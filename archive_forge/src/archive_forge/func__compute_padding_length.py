import torch
import torch.nn.functional as tnn
import tree
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
from keras.src.backend.config import epsilon
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.numpy import expand_dims
from keras.src.backend.torch.numpy import maximum
from keras.src.backend.torch.numpy import where
from keras.src.utils.argument_validation import standardize_tuple
def _compute_padding_length(input_length, kernel_length, stride, dilation_rate=1):
    """Compute padding length along one dimension."""
    total_padding_length = dilation_rate * (kernel_length - 1) - (input_length - 1) % stride
    left_padding = total_padding_length // 2
    right_padding = (total_padding_length + 1) // 2
    return (left_padding, right_padding)