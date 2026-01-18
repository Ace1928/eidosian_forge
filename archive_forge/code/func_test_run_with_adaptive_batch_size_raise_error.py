import numpy as np
import pytest
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from autokeras.utils import utils
def test_run_with_adaptive_batch_size_raise_error():

    def func(**kwargs):
        raise tf.errors.ResourceExhaustedError(0, '', None)
    with pytest.raises(tf.errors.ResourceExhaustedError):
        utils.run_with_adaptive_batch_size(batch_size=64, func=func, x=tf.data.Dataset.from_tensor_slices(np.random.rand(100, 1)).batch(64), validation_data=tf.data.Dataset.from_tensor_slices(np.random.rand(100, 1)).batch(64))