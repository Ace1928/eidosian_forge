import numpy
import pytest
from hypothesis import given
from thinc.api import Padded, Ragged, get_width
from thinc.types import ArgsKwargs
from thinc.util import (
from . import strategies
@given(label_smoothing=strategies.floats(min_value=0.0, max_value=0.5, exclude_max=True))
def test_to_categorical(label_smoothing):
    one_hot = to_categorical(numpy.asarray([1, 2], dtype='i'))
    assert one_hot.shape == (2, 3)
    nc = 5
    shapes = [(1,), (3,), (4, 3), (5, 4, 3), (3, 1), (3, 2, 1)]
    expected_shapes = [(1, nc), (3, nc), (4, 3, nc), (5, 4, 3, nc), (3, 1, nc), (3, 2, 1, nc)]
    labels = [numpy.random.randint(0, nc, shape) for shape in shapes]
    one_hots = [to_categorical(label, nc) for label in labels]
    smooths = [to_categorical(label, nc, label_smoothing=label_smoothing) for label in labels]
    for i in range(len(expected_shapes)):
        label = labels[i]
        one_hot = one_hots[i]
        expected_shape = expected_shapes[i]
        smooth = smooths[i]
        assert one_hot.shape == expected_shape
        assert smooth.shape == expected_shape
        assert numpy.array_equal(one_hot, one_hot.astype(bool))
        assert numpy.all(one_hot.sum(axis=-1) == 1)
        assert numpy.all(numpy.argmax(one_hot, -1).reshape(label.shape) == label)
        assert numpy.all(smooth.argmax(axis=-1) == one_hot.argmax(axis=-1))
        assert numpy.all(numpy.isclose(numpy.sum(smooth, axis=-1), 1.0))
        assert numpy.isclose(numpy.max(smooth), 1 - label_smoothing)
        assert numpy.isclose(numpy.min(smooth), label_smoothing / (smooth.shape[-1] - 1))
    numpy.testing.assert_allclose(to_categorical(numpy.asarray([0, 0, 0]), 1), [[1.0], [1.0], [1.0]])
    numpy.testing.assert_allclose(to_categorical(numpy.asarray([0, 0, 0])), [[1.0], [1.0], [1.0]])
    with pytest.raises(ValueError, match='n_classes should be at least 1'):
        to_categorical(numpy.asarray([0, 0, 0]), 0)
    numpy.testing.assert_allclose(to_categorical(numpy.asarray([0, 1, 0]), 2, label_smoothing=0.01), [[0.99, 0.01], [0.01, 0.99], [0.99, 0.01]])
    numpy.testing.assert_allclose(to_categorical(numpy.asarray([0, 1, 0]), label_smoothing=0.01), [[0.99, 0.01], [0.01, 0.99], [0.99, 0.01]])
    with pytest.raises(ValueError, match='n_classes should be greater than 1.*label smoothing.*but 1'):
        (to_categorical(numpy.asarray([0, 1, 0]), 1, label_smoothing=0.01),)
    with pytest.raises(ValueError, match='n_classes should be greater than 1.*label smoothing.*but 1'):
        (to_categorical(numpy.asarray([0, 0, 0]), label_smoothing=0.01),)
    with pytest.raises(ValueError, match='label_smoothing parameter'):
        to_categorical(numpy.asarray([0, 1, 2, 3, 4]), label_smoothing=0.8)
    with pytest.raises(ValueError, match='label_smoothing parameter'):
        to_categorical(numpy.asarray([0, 1, 2, 3, 4]), label_smoothing=0.88)