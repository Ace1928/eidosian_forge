import os
from typing import Dict, Optional, Union
import numpy as np
import paddle
from safetensors import numpy
def _np2paddle(numpy_dict: Dict[str, np.ndarray], device: str='cpu') -> Dict[str, paddle.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = paddle.to_tensor(v, place=device)
    return numpy_dict