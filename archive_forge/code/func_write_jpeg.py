from enum import Enum
from warnings import warn
import torch
from ..extension import _load_library
from ..utils import _log_api_usage_once
def write_jpeg(input: torch.Tensor, filename: str, quality: int=75):
    """
    Takes an input tensor in CHW layout and saves it in a JPEG file.

    Args:
        input (Tensor[channels, image_height, image_width]): int8 image tensor of ``c``
            channels, where ``c`` must be 1 or 3.
        filename (str): Path to save the image.
        quality (int): Quality of the resulting JPEG file, it must be a number
            between 1 and 100. Default: 75
    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(write_jpeg)
    output = encode_jpeg(input, quality)
    write_file(filename, output)