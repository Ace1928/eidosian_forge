import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def test_noop_transforms(noop_models, ragged_input, padded_input, list_input):
    d_ragged = Ragged(ragged_input.data + 1, ragged_input.lengths)
    d_padded = padded_input.copy()
    d_padded.data += 1
    d_list = [dx + 1 for dx in list_input]
    for model in noop_models:
        print(model.name)
        check_transform_doesnt_change_noop_values(model, padded_input, d_padded)
        check_transform_doesnt_change_noop_values(model, list_input, d_list)
        check_transform_doesnt_change_noop_values(model, ragged_input, d_ragged)