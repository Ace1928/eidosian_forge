from datetime import datetime, timezone, timedelta
import platform
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.units as munits
from matplotlib.category import UnitData
import numpy as np
import pytest
@image_comparison(['plot_pint.png'], style='mpl20', tol=0 if platform.machine() == 'x86_64' else 0.01)
def test_numpy_facade(quantity_converter):
    plt.rcParams['axes.formatter.limits'] = (-7, 7)
    munits.registry[Quantity] = quantity_converter
    y = Quantity(np.linspace(0, 30), 'miles')
    x = Quantity(np.linspace(0, 5), 'hours')
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15)
    ax.plot(x, y, 'tab:blue')
    ax.axhline(Quantity(26400, 'feet'), color='tab:red')
    ax.axvline(Quantity(120, 'minutes'), color='tab:green')
    ax.yaxis.set_units('inches')
    ax.xaxis.set_units('seconds')
    assert quantity_converter.convert.called
    assert quantity_converter.axisinfo.called
    assert quantity_converter.default_units.called