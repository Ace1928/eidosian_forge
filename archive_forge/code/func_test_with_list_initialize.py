import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def test_with_list_initialize(ragged_input, padded_input, list_input):
    for inputs in (ragged_input, padded_input, list_input):
        check_initialize(get_list_model(), inputs)