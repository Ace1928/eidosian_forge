import subprocess
import os
import sys
import pytest
import matplotlib.backends.backend_webagg_core
def test_webagg_core_no_toolbar():
    fm = matplotlib.backends.backend_webagg_core.FigureManagerWebAgg
    assert fm._toolbar2_class is None