from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
class TestPatchDask(TestCase):

    def setUp(self):
        try:
            import dask.dataframe as dd
        except:
            raise SkipTest('Dask not available')
        import hvplot.dask

    def test_dask_series_patched(self):
        import pandas as pd
        import dask.dataframe as dd
        series = pd.Series([0, 1, 2])
        dseries = dd.from_pandas(series, 2)
        self.assertIsInstance(dseries.hvplot, hvPlotTabular)

    def test_dask_dataframe_patched(self):
        import pandas as pd
        import dask.dataframe as dd
        df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['x', 'y'])
        ddf = dd.from_pandas(df, 2)
        self.assertIsInstance(ddf.hvplot, hvPlotTabular)