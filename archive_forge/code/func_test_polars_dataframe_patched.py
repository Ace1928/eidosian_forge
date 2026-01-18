from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
def test_polars_dataframe_patched(self):
    import polars as pl
    pdf = pl.DataFrame({'x': [1, 3, 5], 'y': [2, 4, 6]})
    self.assertIsInstance(pdf.hvplot, hvPlotTabular)