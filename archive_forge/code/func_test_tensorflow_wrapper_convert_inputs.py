import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
@pytest.mark.parametrize('data,n_args,kwargs_keys', [(numpy.zeros((2, 3), dtype='f'), 1, []), ([numpy.zeros((2, 3), dtype='f'), numpy.zeros((2, 3), dtype='f')], 2, []), ((numpy.zeros((2, 3), dtype='f'), numpy.zeros((2, 3), dtype='f')), 2, []), ({'a': numpy.zeros((2, 3), dtype='f'), 'b': numpy.zeros((2, 3), dtype='f')}, 0, ['a', 'b']), (ArgsKwargs((numpy.zeros((2, 3), dtype='f'), numpy.zeros((2, 3), dtype='f')), {'c': numpy.zeros((2, 3), dtype='f')}), 2, ['c'])])
def test_tensorflow_wrapper_convert_inputs(data, n_args, kwargs_keys):
    import tensorflow as tf
    keras_model = tf.keras.Sequential([tf.keras.layers.Dense(12, input_shape=(12,))])
    model = TensorFlowWrapper(keras_model)
    convert_inputs = model.attrs['convert_inputs']
    Y, backprop = convert_inputs(model, data, is_train=True)
    check_input_converters(Y, backprop, data, n_args, kwargs_keys, tf.Tensor)