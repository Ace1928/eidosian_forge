import pytest
import numpy as np
from numpy.core._multiarray_tests import argparse_example_function as func
def test_too_many_positional():
    with pytest.raises(TypeError, match='takes from 2 to 3 positional arguments but 4 were given'):
        func(1, 2, 3, 4)