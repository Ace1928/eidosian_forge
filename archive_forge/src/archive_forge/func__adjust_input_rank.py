import copy
import inspect
import warnings
import tree
from keras.src import backend
from keras.src import ops
from keras.src.backend.common import global_state
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.legacy.saving import saving_utils
from keras.src.legacy.saving import serialization as legacy_serialization
from keras.src.models.model import Model
from keras.src.ops.function import Function
from keras.src.ops.function import make_node_key
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def _adjust_input_rank(self, flat_inputs):
    flat_ref_shapes = [x.shape for x in self._inputs]
    adjusted = []
    for x, ref_shape in zip(flat_inputs, flat_ref_shapes):
        x_rank = len(x.shape)
        ref_rank = len(ref_shape)
        if x_rank == ref_rank:
            adjusted.append(x)
            continue
        if x_rank == ref_rank + 1:
            if x.shape[-1] == 1:
                adjusted.append(ops.squeeze(x, axis=-1))
                continue
        if x_rank == ref_rank - 1:
            if ref_shape[-1] == 1:
                adjusted.append(ops.expand_dims(x, axis=-1))
                continue
        raise ValueError(f'Invalid input shape for input {x}. Expected shape {ref_shape}, but input has incompatible shape {x.shape}')
    for i in range(len(flat_inputs)):
        if hasattr(flat_inputs[i], '_keras_history'):
            adjusted[i]._keras_history = flat_inputs[i]._keras_history
        if hasattr(flat_inputs[i], '_keras_mask'):
            adjusted[i]._keras_mask = flat_inputs[i]._keras_mask
    return adjusted