from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
def test_remove_data_pickle(self):
    results = self.results
    xf = self.xf
    pred_kwds = self.predict_kwds
    pred1 = results.predict(xf, **pred_kwds)
    results.summary()
    results.summary2()
    res, orig_nbytes = check_pickle(results._results)
    results.remove_data()
    pred2 = results.predict(xf, **pred_kwds)
    if isinstance(pred1, pd.Series) and isinstance(pred2, pd.Series):
        assert_series_equal(pred1, pred2)
    elif isinstance(pred1, pd.DataFrame) and isinstance(pred2, pd.DataFrame):
        assert pred1.equals(pred2)
    else:
        np.testing.assert_equal(pred2, pred1)
    res, nbytes = check_pickle(results._results)
    self.res = res
    msg = 'pickle length not %d < %d' % (nbytes, orig_nbytes)
    assert nbytes < orig_nbytes * self.reduction_factor, msg
    pred3 = results.predict(xf, **pred_kwds)
    if isinstance(pred1, pd.Series) and isinstance(pred3, pd.Series):
        assert_series_equal(pred1, pred3)
    elif isinstance(pred1, pd.DataFrame) and isinstance(pred3, pd.DataFrame):
        assert pred1.equals(pred3)
    else:
        np.testing.assert_equal(pred3, pred1)