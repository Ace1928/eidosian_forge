import datetime
import json
import pandas as pd
from yfinance import utils, const
from yfinance.data import YfData
from yfinance.exceptions import YFinanceException, YFNotImplementedError
@utils.log_indent_decorator
def _fetch_time_series(self, name, timescale, proxy=None):
    allowed_names = ['income', 'balance-sheet', 'cash-flow']
    allowed_timescales = ['yearly', 'quarterly']
    if name not in allowed_names:
        raise ValueError(f'Illegal argument: name must be one of: {allowed_names}')
    if timescale not in allowed_timescales:
        raise ValueError(f'Illegal argument: timescale must be one of: {allowed_timescales}')
    try:
        statement = self._create_financials_table(name, timescale, proxy)
        if statement is not None:
            return statement
    except YFinanceException as e:
        utils.get_yf_logger().error(f'{self._symbol}: Failed to create {name} financials table for reason: {e}')
    return pd.DataFrame()