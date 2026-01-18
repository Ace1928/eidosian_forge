import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from ..utils import _log_api_usage_once
from . import functional as F
from .functional import _interpolation_modes_from_int, InterpolationMode

        Args:
            tensor (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        