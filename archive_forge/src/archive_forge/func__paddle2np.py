import os
from typing import Dict, Optional, Union
import numpy as np
import paddle
from safetensors import numpy
def _paddle2np(paddle_dict: Dict[str, paddle.Tensor]) -> Dict[str, np.array]:
    for k, v in paddle_dict.items():
        paddle_dict[k] = v.detach().cpu().numpy()
    return paddle_dict