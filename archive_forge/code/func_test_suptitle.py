import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def test_suptitle():
    fig, ax = plt.subplots(tight_layout=True)
    st = fig.suptitle('foo')
    t = ax.set_title('bar')
    fig.canvas.draw()
    assert st.get_window_extent().y0 > t.get_window_extent().y1