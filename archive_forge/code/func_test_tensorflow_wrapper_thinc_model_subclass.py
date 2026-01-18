import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
def test_tensorflow_wrapper_thinc_model_subclass(tf_model):

    class CustomModel(Model):

        def fn(self):
            return 1337
    model = TensorFlowWrapper(tf_model, model_class=CustomModel)
    assert isinstance(model, CustomModel)
    assert model.fn() == 1337