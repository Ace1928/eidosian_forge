import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def test_pylab_integration():
    IPython = pytest.importorskip('IPython')
    mpl.testing.subprocess_run_helper(IPython.start_ipython, '--pylab', '-c', ';'.join(('import matplotlib.pyplot as plt', 'assert plt._REPL_DISPLAYHOOK == plt._ReplDisplayHook.IPYTHON')), timeout=60)