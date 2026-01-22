import os
from typing import Dict, Optional, Union
import numpy as np
import tensorflow as tf
from safetensors import numpy, safe_open
def _tf2np(tf_dict: Dict[str, tf.Tensor]) -> Dict[str, np.array]:
    for k, v in tf_dict.items():
        tf_dict[k] = v.numpy()
    return tf_dict