import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
@pytest.fixture
def allow_exact_matches_and_tolerance(self):
    df = pd.DataFrame([['20160525 13:30:00.023', 'MSFT', '51.95', '75', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.038', 'MSFT', '51.95', '155', 'NASDAQ', '51.95', '51.95'], ['20160525 13:30:00.048', 'GOOG', '720.77', '100', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.92', '100', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '200', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '300', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '600', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.048', 'GOOG', '720.93', '44', 'NASDAQ', '720.5', '720.93'], ['20160525 13:30:00.074', 'AAPL', '98.67', '478343', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.67', '478343', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.66', '6', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.65', '30', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.65', '75', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.65', '20', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.65', '35', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.65', '10', 'NASDAQ', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.55', '6', 'ARCA', np.nan, np.nan], ['20160525 13:30:00.075', 'AAPL', '98.55', '6', 'ARCA', np.nan, np.nan], ['20160525 13:30:00.076', 'AAPL', '98.56', '1000', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '200', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '300', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '400', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '600', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.076', 'AAPL', '98.56', '200', 'ARCA', '98.55', '98.56'], ['20160525 13:30:00.078', 'MSFT', '51.95', '783', 'NASDAQ', '51.95', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.95', '100', 'NASDAQ', '51.95', '51.95'], ['20160525 13:30:00.078', 'MSFT', '51.95', '100', 'NASDAQ', '51.95', '51.95']], columns='time,ticker,price,quantity,marketCenter,bid,ask'.split(','))
    df['price'] = df['price'].astype('float64')
    df['quantity'] = df['quantity'].astype('int64')
    df['bid'] = df['bid'].astype('float64')
    df['ask'] = df['ask'].astype('float64')
    return self.prep_data(df)