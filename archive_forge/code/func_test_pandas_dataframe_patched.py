from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
def test_pandas_dataframe_patched(self):
    import pandas as pd
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['x', 'y'])
    self.assertIsInstance(df.hvplot, hvPlotTabular)