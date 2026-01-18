from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@image_comparison(['scatter_post_alpha.png'], remove_text=True, style='default')
def test_scatter_post_alpha():
    fig, ax = plt.subplots()
    sc = ax.scatter(range(5), range(5), c=range(5))
    sc.set_alpha(0.1)