import inspect
import re
import warnings
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import (
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from warnings import warn
def test_parallel_warning():

    @run_in_parallel()
    def change_state_warns_fails():
        warn('Test warning for test parallel', stacklevel=2)
    with expected_warnings(['Test warning for test parallel']):
        change_state_warns_fails()

    @run_in_parallel(warnings_matching=['Test warning for test parallel'])
    def change_state_warns_passes():
        warn('Test warning for test parallel', stacklevel=2)
    change_state_warns_passes()