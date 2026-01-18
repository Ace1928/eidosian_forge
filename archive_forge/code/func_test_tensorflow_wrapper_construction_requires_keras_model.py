import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
def test_tensorflow_wrapper_construction_requires_keras_model():
    import tensorflow as tf
    keras_model = tf.keras.Sequential([tf.keras.layers.Dense(12, input_shape=(12,))])
    assert isinstance(TensorFlowWrapper(keras_model), Model)
    with pytest.raises(ValueError):
        TensorFlowWrapper(Linear(2, 3))