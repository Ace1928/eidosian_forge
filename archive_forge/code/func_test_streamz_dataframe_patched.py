from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
def test_streamz_dataframe_patched(self):
    from streamz.dataframe import Random
    random_df = Random()
    self.assertIsInstance(random_df.hvplot, hvPlotTabular)