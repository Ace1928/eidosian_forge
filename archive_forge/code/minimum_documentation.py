import tensorflow.compat.v2 as tf
from keras.src.layers.merging.base_merge import _Merge
from tensorflow.python.util.tf_export import keras_export
Functional interface to the `Minimum` layer.

    Args:
        inputs: A list of input tensors.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the element-wise minimum of the inputs.
    