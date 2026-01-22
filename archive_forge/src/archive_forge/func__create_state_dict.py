import contextlib
import copy
from io import BytesIO
from typing import Any, Dict, List, Optional
import catalogue
import numpy
from ..backends import Ops, get_current_ops
from ..compat import cupy, h5py
from ..compat import tensorflow as tf
from ..optimizers import Optimizer
from ..types import ArgsKwargs, ArrayXd
from ..util import get_array_module
from .shim import Shim
def _create_state_dict(self):
    state_dict = {}
    for layer in self._model.layers:
        for weight in layer.weights:
            state_dict[weight.name] = weight.numpy()
    return state_dict