import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def test_with_ragged_forward(ragged_input, padded_input, list_input, ragged_data_input):
    for inputs in (ragged_input, padded_input, list_input, ragged_data_input):
        checker = get_data_checker(inputs)
        model = get_ragged_model()
        check_transform_produces_correct_output_type_forward(model, inputs, checker)