import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
def test_warning_removed_param(self):
    match = '.*`old[01]` is deprecated since version 0\\.10 and will be removed in 0\\.12.* see the documentation of .*_func_deprecated_params`.'
    with pytest.warns(FutureWarning, match=match):
        assert _func_deprecated_params(1, 2) == (1, DEPRECATED, DEPRECATED, None)
    with pytest.warns(FutureWarning, match=match):
        assert _func_deprecated_params(1, 2, 3) == (1, DEPRECATED, DEPRECATED, None)
    with pytest.warns(FutureWarning, match=match):
        assert _func_deprecated_params(1, old0=2) == (1, DEPRECATED, DEPRECATED, None)
    with pytest.warns(FutureWarning, match=match):
        assert _func_deprecated_params(1, old1=2) == (1, DEPRECATED, DEPRECATED, None)
    with warnings.catch_warnings(record=True) as record:
        assert _func_deprecated_params(1, arg1=3) == (1, DEPRECATED, DEPRECATED, 3)
    assert len(record) == 0