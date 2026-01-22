import sys
import pytest
import numpy as np
from typing import NamedTuple
from numpy.testing import assert_allclose
from scipy.special import hyp2f1
from scipy.special._testutils import check_version, MissingModule
def _get_test_parameters(self, test_method):
    """Get pytest.mark parameters for a test in this class."""
    return [case.values[0] for mark in test_method.pytestmark if mark.name == 'parametrize' for case in mark.args[1]]