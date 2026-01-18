import functools
import itertools
import platform
import pytest
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import (MouseButton, MouseEvent,
from matplotlib import cm
from matplotlib import colors as mcolors, patches as mpatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
@image_comparison(baseline_images=['panecolor_rcparams.png'], remove_text=True, style='mpl20')
def test_panecolor_rcparams():
    with plt.rc_context({'axes3d.xaxis.panecolor': 'r', 'axes3d.yaxis.panecolor': 'g', 'axes3d.zaxis.panecolor': 'b'}):
        fig = plt.figure(figsize=(1, 1))
        fig.add_subplot(projection='3d')