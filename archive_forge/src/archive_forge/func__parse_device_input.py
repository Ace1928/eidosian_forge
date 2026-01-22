import contextlib
import os
import ml_dtypes
import numpy as np
import torch
import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.config import floatx
from keras.src.utils.nest import pack_sequence_as
def _parse_device_input(device_name):
    if isinstance(device_name, str):
        device_name = device_name.lower()
        if 'gpu' in device_name:
            device_name = device_name.replace('gpu', 'cuda')
    else:
        raise ValueError(f"Invalid value for argument `device_name`. Expected a string like 'gpu:0' or 'cpu'. Received: device_name='{device_name}'")
    return device_name