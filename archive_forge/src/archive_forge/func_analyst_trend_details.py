import pandas as pd
from yfinance.data import YfData
from yfinance.exceptions import YFNotImplementedError
@property
def analyst_trend_details(self) -> pd.DataFrame:
    if self._analyst_trend_details is None:
        raise YFNotImplementedError('analyst_trend_details')
    return self._analyst_trend_details