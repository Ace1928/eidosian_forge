import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
def test_warning_location(self):
    with pytest.warns(FutureWarning) as records:
        _func_deprecated_params(1, old0=2, old1=2)
        testing.assert_stacklevel(records)
    assert len(records) == 2