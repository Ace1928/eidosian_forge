import numpy
import pytest
from thinc.api import NumpyOps, Ragged, registry, strings2arrays
from ..util import get_data_checker
def test_padded2list(padded_data, list_data):
    check_transform('padded2list.v1', padded_data, list_data)