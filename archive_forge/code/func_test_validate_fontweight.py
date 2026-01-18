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
@pytest.mark.parametrize('weight, parsed_weight', [('bold', 'bold'), ('BOLD', ValueError), (100, 100), ('100', 100), (np.array(100), 100), (20.6, 20), ('20.6', ValueError), ([100], ValueError)])
def test_validate_fontweight(weight, parsed_weight):
    if parsed_weight is ValueError:
        with pytest.raises(ValueError):
            validate_fontweight(weight)
    else:
        assert validate_fontweight(weight) == parsed_weight