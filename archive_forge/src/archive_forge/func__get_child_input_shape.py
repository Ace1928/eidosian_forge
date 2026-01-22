from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.core.wrapper import Wrapper
from keras.src.layers.layer import Layer
def _get_child_input_shape(self, input_shape):
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
        raise ValueError(f'`TimeDistributed` Layer should be passed an `input_shape` with at least 3 dimensions, received: {input_shape}')
    return (input_shape[0], *input_shape[2:])