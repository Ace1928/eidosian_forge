import os
from typing import Dict, Optional, Union
import numpy as np
import mlx.core as mx
from safetensors import numpy, safe_open
def _np2mx(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, mx.array]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = mx.array(v)
    return numpy_dict