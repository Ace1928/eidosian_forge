from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
def test_pandas_series_patched(self):
    import pandas as pd
    series = pd.Series([0, 1, 2])
    self.assertIsInstance(series.hvplot, hvPlotTabular)