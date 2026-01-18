import numpy as np
import pytest
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from autokeras.utils import utils
def test_validate_num_inputs_error():
    with pytest.raises(ValueError) as info:
        utils.validate_num_inputs([1, 2, 3], 2)
    assert 'Expected 2 elements in the inputs list' in str(info.value)