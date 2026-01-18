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
@pytest.mark.parametrize('validator, arg, exception_type', generate_validator_testcases(False))
def test_validator_invalid(validator, arg, exception_type):
    with pytest.raises(exception_type):
        validator(arg)