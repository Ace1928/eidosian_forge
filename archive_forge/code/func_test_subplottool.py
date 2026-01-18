import copy
import importlib
import os
import signal
import sys
from datetime import date, datetime
from unittest import mock
import pytest
import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib import _c_internal_utils
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_subplottool():
    fig, ax = plt.subplots()
    with mock.patch('matplotlib.backends.qt_compat._exec', lambda obj: None):
        fig.canvas.manager.toolbar.configure_subplots()