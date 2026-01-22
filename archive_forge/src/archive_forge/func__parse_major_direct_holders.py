import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import _BASE_URL_
from yfinance.exceptions import YFinanceDataException
def _parse_major_direct_holders(self, data):
    holders = data['holders']
    for owner in holders:
        for k, v in owner.items():
            owner[k] = self._parse_raw_values(v)
        del owner['maxAge']
    df = pd.DataFrame(holders)
    if not df.empty:
        df['reportDate'] = pd.to_datetime(df['reportDate'], unit='s')
        df.rename(columns={'reportDate': 'Date Reported', 'organization': 'Holder', 'positionDirect': 'Shares', 'valueDirect': 'Value'}, inplace=True)
    self._major_direct_holders = df