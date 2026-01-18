import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
@image_comparison(['tight_layout8'])
def test_tight_layout8():
    """Test automatic use of tight_layout."""
    fig = plt.figure()
    fig.set_layout_engine(layout='tight', pad=0.1)
    ax = fig.add_subplot()
    example_plot(ax, fontsize=24)
    fig.draw_without_rendering()