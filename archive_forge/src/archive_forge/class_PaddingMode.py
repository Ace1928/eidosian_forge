import warnings
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
from .image_utils import (
from .utils import ExplicitEnum, TensorType, is_jax_tensor, is_tf_tensor, is_torch_tensor
from .utils.import_utils import (
class PaddingMode(ExplicitEnum):
    """
    Enum class for the different padding modes to use when padding images.
    """
    CONSTANT = 'constant'
    REFLECT = 'reflect'
    REPLICATE = 'replicate'
    SYMMETRIC = 'symmetric'