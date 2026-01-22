import numpy as np
import torch
from contextlib import contextmanager
from torch.testing._internal.common_utils import TEST_WITH_ASAN, TEST_WITH_TSAN, TEST_WITH_UBSAN, IS_PPC, IS_MACOS, IS_WINDOWS
def _conv_output_shape(input_size, kernel_size, padding, stride, dilation, output_padding=0):
    """Computes the output shape given convolution parameters."""
    return np.floor((input_size + 2 * padding - kernel_size - (kernel_size - 1) * (dilation - 1)) / stride) + 2 * output_padding + 1