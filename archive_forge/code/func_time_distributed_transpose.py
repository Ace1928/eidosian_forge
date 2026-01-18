from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.core.wrapper import Wrapper
from keras.src.layers.layer import Layer
def time_distributed_transpose(data):
    """Swaps the timestep and batch dimensions of a tensor."""
    axes = [1, 0, *range(2, len(data.shape))]
    return ops.transpose(data, axes=axes)