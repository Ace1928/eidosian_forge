import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def test_stackplot_smoke():
    plt.stackplot([1, 2, 3], [1, 2, 3])