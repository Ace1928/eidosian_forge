from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
class TestPatchPandas(TestCase):

    def setUp(self):
        import hvplot.pandas

    def test_pandas_series_patched(self):
        import pandas as pd
        series = pd.Series([0, 1, 2])
        self.assertIsInstance(series.hvplot, hvPlotTabular)

    def test_pandas_dataframe_patched(self):
        import pandas as pd
        df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['x', 'y'])
        self.assertIsInstance(df.hvplot, hvPlotTabular)