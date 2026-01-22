import os
from typing import Dict, Optional, Union
import numpy as np
import mlx.core as mx
from safetensors import numpy, safe_open
def _mx2np(mx_dict: Dict[str, mx.array]) -> Dict[str, np.array]:
    new_dict = {}
    for k, v in mx_dict.items():
        new_dict[k] = np.asarray(v)
    return new_dict