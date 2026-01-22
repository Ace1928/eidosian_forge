import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
def _ensure_format_supported(self, image):
    if not isinstance(image, (PIL.Image.Image, np.ndarray)) and (not is_torch_tensor(image)):
        raise ValueError(f'Got type {type(image)} which is not supported, only `PIL.Image.Image`, `np.array` and `torch.Tensor` are.')