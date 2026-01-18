from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt, style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION
def test_use():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            assert mpl.rcParams[PARAM] == VALUE