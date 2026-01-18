import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
def test_tensorflow_wrapper_serialize_model_subclass(X, Y, input_size, n_classes, answer):
    import tensorflow as tf
    input_shape = (1, input_size)
    ops = get_current_ops()

    @keras_subclass('foo.v1', X=ops.alloc2f(*input_shape), Y=to_categorical(ops.asarray1i([1]), n_classes=n_classes), input_shape=input_shape)
    class CustomKerasModel(tf.keras.Model):

        def __init__(self, **kwargs):
            super(CustomKerasModel, self).__init__(**kwargs)
            self.in_dense = tf.keras.layers.Dense(12, name='in_dense', input_shape=input_shape)
            self.out_dense = tf.keras.layers.Dense(n_classes, name='out_dense', activation='softmax')

        def call(self, inputs) -> tf.Tensor:
            x = self.in_dense(inputs)
            return self.out_dense(x)
    model = TensorFlowWrapper(CustomKerasModel())
    optimizer = Adam()
    for i in range(50):
        guesses, backprop = model(X, is_train=True)
        guesses = ops.asarray(guesses)
        d_guesses = (guesses - Y) / guesses.shape[0]
        backprop(d_guesses)
        model.finish_update(optimizer)
    predicted = model.predict(X).argmax()
    assert predicted == answer
    model.from_bytes(model.to_bytes())
    assert model.predict(X).argmax() == answer