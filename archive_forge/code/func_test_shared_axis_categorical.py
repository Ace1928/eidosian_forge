from datetime import datetime, timezone, timedelta
import platform
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.units as munits
from matplotlib.category import UnitData
import numpy as np
import pytest
def test_shared_axis_categorical():
    d1 = {'a': 1, 'b': 2}
    d2 = {'a': 3, 'b': 4}
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.plot(d1.keys(), d1.values())
    ax2.plot(d2.keys(), d2.values())
    ax1.xaxis.set_units(UnitData(['c', 'd']))
    assert 'c' in ax2.xaxis.get_units()._mapping.keys()