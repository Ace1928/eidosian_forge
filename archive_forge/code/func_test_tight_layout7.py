import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
@image_comparison(['tight_layout7'], tol=1.9)
def test_tight_layout7():
    fontsize = 24
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Left Title', loc='left', fontsize=fontsize)
    ax.set_title('Right Title', loc='right', fontsize=fontsize)
    plt.tight_layout()