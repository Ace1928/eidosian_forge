import numpy as np
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.util.annotations import PublicAPI
Returns the NumPy dtype of the given tensor or array.