import types
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.metrics import base_metric
Computes and returns the scalar metric value.

        **Note:** This function is executed outside of the TensorFlow graph
         on the CPU host. This means any TensorFlow ops run in this method
         are run eagerly.

        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.

        Returns:
            A Python scalar.
        