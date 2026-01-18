import numpy
import pytest
from thinc.api import NumpyOps, Ragged, registry, strings2arrays
from ..util import get_data_checker
def test_ragged2list(ragged_data, list_data):
    check_transform('ragged2list.v1', ragged_data, list_data)