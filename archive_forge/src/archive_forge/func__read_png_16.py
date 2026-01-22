from enum import Enum
from warnings import warn
import torch
from ..extension import _load_library
from ..utils import _log_api_usage_once
def _read_png_16(path: str, mode: ImageReadMode=ImageReadMode.UNCHANGED) -> torch.Tensor:
    data = read_file(path)
    return torch.ops.image.decode_png(data, mode.value, True)