import datetime
import json
import pandas as pd
from yfinance import utils, const
from yfinance.data import YfData
from yfinance.exceptions import YFinanceException, YFNotImplementedError
class Fundamentals:

    def __init__(self, data: YfData, symbol: str, proxy=None):
        self._data = data
        self._symbol = symbol
        self.proxy = proxy
        self._earnings = None
        self._financials = None
        self._shares = None
        self._financials_data = None
        self._fin_data_quote = None
        self._basics_already_scraped = False
        self._financials = Financials(data, symbol)

    @property
    def financials(self) -> 'Financials':
        return self._financials

    @property
    def earnings(self) -> dict:
        if self._earnings is None:
            raise YFNotImplementedError('earnings')
        return self._earnings

    @property
    def shares(self) -> pd.DataFrame:
        if self._shares is None:
            raise YFNotImplementedError('shares')
        return self._shares