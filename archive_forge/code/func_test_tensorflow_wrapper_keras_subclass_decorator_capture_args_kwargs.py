import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
def test_tensorflow_wrapper_keras_subclass_decorator_capture_args_kwargs(X, Y, input_size, n_classes, answer):
    import tensorflow as tf

    @keras_subclass('TestModel', X=numpy.array([0.0, 0.0]), Y=numpy.array([0.5]), input_shape=(2,))
    class TestModel(tf.keras.Model):

        def __init__(self, custom=False, **kwargs):
            super().__init__(self)
            assert custom is True
            assert kwargs.get('other', None) is not None

        def call(self, inputs):
            return inputs
    model = TensorFlowWrapper(TestModel(True, other=1337))
    assert hasattr(model.shims[0]._model, 'eg_args')
    args_kwargs = model.shims[0]._model.eg_args
    assert True in args_kwargs.args
    assert 'other' in args_kwargs.kwargs
    obj = {}
    obj['key'] = obj
    with pytest.raises(ValueError):
        TensorFlowWrapper(TestModel(True, other=obj))
    model = model.from_bytes(model.to_bytes())