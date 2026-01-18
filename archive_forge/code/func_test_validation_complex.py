import pytest
from thinc.api import (
from thinc.util import DataValidationError, data_validation
def test_validation_complex():
    good_model = chain(list2ragged(), reduce_sum(), Relu(12, dropout=0.5), Relu(1))
    X = [good_model.ops.xp.zeros((4, 75), dtype='f')]
    Y = good_model.ops.xp.zeros((1,), dtype='f')
    good_model.initialize(X, Y)
    good_model.predict(X)
    bad_model = chain(list2ragged(), reduce_sum(), Relu(12, dropout=0.5), ParametricAttention(12), Relu(1))
    with data_validation(True):
        with pytest.raises(DataValidationError):
            bad_model.initialize(X, Y)