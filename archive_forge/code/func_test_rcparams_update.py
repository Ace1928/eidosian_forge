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
def test_rcparams_update():
    rc = mpl.RcParams({'figure.figsize': (3.5, 42)})
    bad_dict = {'figure.figsize': (3.5, 42, 1)}
    with pytest.raises(ValueError):
        rc.update(bad_dict)