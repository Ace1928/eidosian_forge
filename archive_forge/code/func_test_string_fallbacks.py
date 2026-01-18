import pytest
import numpy as np
from numpy.core._multiarray_tests import argparse_example_function as func
def test_string_fallbacks():
    arg2 = np.str_('arg2')
    missing_arg = np.str_('missing_arg')
    func(1, **{arg2: 3})
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'missing_arg'"):
        func(2, **{missing_arg: 3})