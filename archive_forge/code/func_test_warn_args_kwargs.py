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
def test_warn_args_kwargs(self):
    fig, axs = plt.subplots(1, 2)
    lines = axs[0].plot(range(10))
    lines2 = axs[1].plot(np.arange(10) * 2.0)
    with pytest.warns(UserWarning) as record:
        fig.legend((lines, lines2), labels=('a', 'b'))
    assert len(record) == 1
    assert str(record[0].message) == 'You have mixed positional and keyword arguments, some input may be discarded.'