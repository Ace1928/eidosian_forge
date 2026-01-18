import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def test_pyplot_box():
    fig, ax = plt.subplots()
    plt.box(False)
    assert not ax.get_frame_on()
    plt.box(True)
    assert ax.get_frame_on()
    plt.box()
    assert not ax.get_frame_on()
    plt.box()
    assert ax.get_frame_on()