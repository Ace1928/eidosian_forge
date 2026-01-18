import copy
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock
from cycler import cycler, Cycler
import pytest
import matplotlib as mpl
from matplotlib import _api, _c_internal_utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.rcsetup import (
@pytest.mark.parametrize('validator, arg, target', generate_validator_testcases(True))
def test_validator_valid(validator, arg, target):
    res = validator(arg)
    if isinstance(target, np.ndarray):
        np.testing.assert_equal(res, target)
    elif not isinstance(target, Cycler):
        assert res == target
    else:
        assert list(res) == list(target)