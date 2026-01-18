from __future__ import annotations
import io
from pathlib import Path
import platform
import re
import shlex
from xml.etree import ElementTree as ET
from typing import Any
import numpy as np
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
from matplotlib import mathtext, _mathtext
def test_single_minus_sign():
    fig = plt.figure()
    fig.text(0.5, 0.5, '$-$')
    fig.canvas.draw()
    t = np.asarray(fig.canvas.renderer.buffer_rgba())
    assert (t != 255).any()