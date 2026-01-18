import collections
import platform
from unittest import mock
import warnings
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend as mlegend
from matplotlib import _api, rc_context
from matplotlib.font_manager import FontProperties
def test_reverse_legend_handles_and_labels():
    """Check that the legend handles and labels are reversed."""
    fig, ax = plt.subplots()
    x = 1
    y = 1
    labels = ['First label', 'Second label', 'Third label']
    markers = ['.', ',', 'o']
    ax.plot(x, y, markers[0], label=labels[0])
    ax.plot(x, y, markers[1], label=labels[1])
    ax.plot(x, y, markers[2], label=labels[2])
    leg = ax.legend(reverse=True)
    actual_labels = [t.get_text() for t in leg.get_texts()]
    actual_markers = [h.get_marker() for h in leg.legend_handles]
    assert actual_labels == list(reversed(labels))
    assert actual_markers == list(reversed(markers))