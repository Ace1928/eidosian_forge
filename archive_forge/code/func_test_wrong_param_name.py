import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
def test_wrong_param_name(self):
    with pytest.raises(ValueError, match="'old' is not in list"):

        @deprecate_parameter('old', start_version='0.10', stop_version='0.12')
        def foo(arg0):
            pass
    with pytest.raises(ValueError, match="'new' is not in list"):

        @deprecate_parameter('old', new_name='new', start_version='0.10', stop_version='0.12')
        def bar(arg0, old, arg1):
            pass