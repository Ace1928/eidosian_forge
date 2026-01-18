from datetime import (
import subprocess
import sys
import numpy as np
import pytest
import pandas._config.config as cf
from pandas._libs.tslibs import to_offset
from pandas import (
import pandas._testing as tm
from pandas.plotting import (
from pandas.tseries.offsets import (
def test_registry_resets(self):
    units = pytest.importorskip('matplotlib.units')
    dates = pytest.importorskip('matplotlib.dates')
    original = dict(units.registry)
    try:
        units.registry.clear()
        date_converter = dates.DateConverter()
        units.registry[datetime] = date_converter
        units.registry[date] = date_converter
        register_matplotlib_converters()
        assert units.registry[date] is not date_converter
        deregister_matplotlib_converters()
        assert units.registry[date] is date_converter
    finally:
        units.registry.clear()
        for k, v in original.items():
            units.registry[k] = v