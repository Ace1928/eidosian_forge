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
def _load_weights_from_state_dict(self, state_dict: Optional[Dict[str, ArrayXd]]=None):
    if state_dict is None:
        state_dict = self._create_state_dict()
    for layer in self._model.layers:
        current_layer_weights = []
        for weight in layer.weights:
            current_layer_weights.append(state_dict[weight.name])
        layer.set_weights(current_layer_weights)