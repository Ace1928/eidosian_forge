from datetime import datetime, timezone, timedelta
import platform
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.units as munits
from matplotlib.category import UnitData
import numpy as np
import pytest
def test_plot_kernel():
    kernel = Kernel([1, 2, 3, 4, 5])
    plt.plot(kernel)