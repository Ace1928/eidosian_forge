import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def test_with_padded_initialize(ragged_input, padded_input, list_input, padded_data_input):
    for inputs in (ragged_input, padded_input, list_input, padded_data_input):
        check_initialize(get_padded_model(), inputs)