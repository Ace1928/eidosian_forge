import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
def test_wrong_call_signature(self):
    """Check that normal errors for faulty calls are unchanged."""
    with pytest.raises(TypeError, match=".* required positional argument\\: 'arg0'"):
        _func_replace_params()
    with pytest.warns(FutureWarning, match='.*`old[0,1]` is deprecated'):
        with pytest.raises(TypeError, match=".* multiple values for argument 'old0'"):
            _func_deprecated_params(1, 2, old0=2)