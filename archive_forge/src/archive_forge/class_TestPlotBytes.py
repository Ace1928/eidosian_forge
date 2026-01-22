import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
class TestPlotBytes:
    bytes_cases = [('string list', ['a', 'b', 'c']), ('bytes list', [b'a', b'b', b'c']), ('bytes ndarray', np.array([b'a', b'b', b'c']))]
    bytes_ids, bytes_data = zip(*bytes_cases)

    @pytest.mark.parametrize('plotter', PLOT_LIST, ids=PLOT_IDS)
    @pytest.mark.parametrize('bdata', bytes_data, ids=bytes_ids)
    def test_plot_bytes(self, plotter, bdata):
        ax = plt.figure().subplots()
        counts = np.array([4, 6, 5])
        plotter(ax, bdata, counts)
        axis_test(ax.xaxis, bdata)