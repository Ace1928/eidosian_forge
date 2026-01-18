import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
def test_tensorflow_wrapper_train_overfits(model, X, Y, answer):
    optimizer = Adam()
    ops = get_current_ops()
    for i in range(100):
        guesses, backprop = model(X, is_train=True)
        guesses = ops.asarray(guesses)
        d_guesses = (guesses - Y) / guesses.shape[0]
        backprop(d_guesses)
        model.finish_update(optimizer)
    predicted = model.predict(X).argmax()
    assert predicted == answer