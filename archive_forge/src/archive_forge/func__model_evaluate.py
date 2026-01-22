import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src import callbacks as callbacks_lib
from keras.src.engine import sequential
from keras.src.layers import core as core_layers
from keras.src.layers.preprocessing import string_lookup
from keras.src.optimizers.legacy import gradient_descent
from keras.src.utils import dataset_creator
from tensorflow.python.platform import tf_logging as logging
def _model_evaluate(self, strategy, steps_per_execution=1, x=None, y=None, batch_size=None, steps=10, run_eagerly=False, with_normalization_layer=False, callbacks=None, use_dataset_creator=True):
    if callbacks is None:
        callbacks = []
    model, default_callbacks = self._model_compile(strategy, steps_per_execution, run_eagerly, with_normalization_layer)
    callbacks += default_callbacks

    def dataset_fn(input_context):
        del input_context
        x = tf.random.uniform((10, 10))
        y = tf.random.uniform((10, 1))
        return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat().batch(8)
    if x is None:
        if use_dataset_creator:
            x = dataset_creator.DatasetCreator(dataset_fn)
        else:
            x = dataset_fn(None)
    model.evaluate(x=x, y=y, steps=steps, callbacks=callbacks, batch_size=batch_size)
    return model