import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
def test_tensorflow_wrapper_to_from_disk(model, X, Y, answer):
    with make_tempdir() as tmp_path:
        model_file = tmp_path / 'model.h5'
        model.to_disk(model_file)
        another_model = model.from_disk(model_file)
        assert another_model is not None