import collections
import tree
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.config import backend
from keras.src.ops.operation import Operation
from keras.src.utils.nest import pack_sequence_as
def _assert_input_compatibility(self, inputs):
    try:
        tree.assert_same_structure(inputs, self._inputs_struct, check_types=False)
    except ValueError:
        raise ValueError(f'Function was called with an invalid input structure. Expected input structure: {self._inputs_struct}\nReceived input structure: {inputs}')
    for x, x_ref in zip(tree.flatten(inputs), self._inputs):
        if len(x.shape) != len(x_ref.shape):
            raise ValueError(f"{self.__class__.__name__} was passed incompatible inputs. For input '{x_ref.name}', expected shape {x_ref.shape}, but received instead a tensor with shape {x.shape}.")
        for dim, ref_dim in zip(x.shape, x_ref.shape):
            if ref_dim is not None and dim is not None:
                if dim != ref_dim:
                    raise ValueError(f"{self.__class__.__name__} was passed incompatible inputs. For input '{x_ref.name}', expected shape {x_ref.shape}, but received instead a tensor with shape {x.shape}.")