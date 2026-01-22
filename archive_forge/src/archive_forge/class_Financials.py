import datetime
import json
import pandas as pd
from yfinance import utils, const
from yfinance.data import YfData
from yfinance.exceptions import YFinanceException, YFNotImplementedError
class Financials:

    def __init__(self, data: YfData, symbol: str):
        self._data = data
        self._symbol = symbol
        self._income_time_series = {}
        self._balance_sheet_time_series = {}
        self._cash_flow_time_series = {}

    def get_income_time_series(self, freq='yearly', proxy=None) -> pd.DataFrame:
        res = self._income_time_series
        if freq not in res:
            res[freq] = self._fetch_time_series('income', freq, proxy)
        return res[freq]

    def get_balance_sheet_time_series(self, freq='yearly', proxy=None) -> pd.DataFrame:
        res = self._balance_sheet_time_series
        if freq not in res:
            res[freq] = self._fetch_time_series('balance-sheet', freq, proxy)
        return res[freq]

    def get_cash_flow_time_series(self, freq='yearly', proxy=None) -> pd.DataFrame:
        res = self._cash_flow_time_series
        if freq not in res:
            res[freq] = self._fetch_time_series('cash-flow', freq, proxy)
        return res[freq]

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

    def _create_financials_table(self, name, timescale, proxy):
        if name == 'income':
            name = 'financials'
        keys = const.fundamentals_keys[name]
        try:
            return self.get_financials_time_series(timescale, keys, proxy)
        except Exception:
            pass

    def get_financials_time_series(self, timescale, keys: list, proxy=None) -> pd.DataFrame:
        timescale_translation = {'yearly': 'annual', 'quarterly': 'quarterly'}
        timescale = timescale_translation[timescale]
        ts_url_base = f'https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{self._symbol}?symbol={self._symbol}'
        url = ts_url_base + '&type=' + ','.join([timescale + k for k in keys])
        start_dt = datetime.datetime(2016, 12, 31)
        end = pd.Timestamp.utcnow().ceil('D')
        url += f'&period1={int(start_dt.timestamp())}&period2={int(end.timestamp())}'
        json_str = self._data.cache_get(url=url, proxy=proxy).text
        json_data = json.loads(json_str)
        data_raw = json_data['timeseries']['result']
        for d in data_raw:
            del d['meta']
        timestamps = set()
        data_unpacked = {}
        for x in data_raw:
            for k in x.keys():
                if k == 'timestamp':
                    timestamps.update(x[k])
                else:
                    data_unpacked[k] = x[k]
        timestamps = sorted(list(timestamps))
        dates = pd.to_datetime(timestamps, unit='s')
        df = pd.DataFrame(columns=dates, index=list(data_unpacked.keys()))
        for k, v in data_unpacked.items():
            if df is None:
                df = pd.DataFrame(columns=dates, index=[k])
            df.loc[k] = {pd.Timestamp(x['asOfDate']): x['reportedValue']['raw'] for x in v}
        df.index = df.index.str.replace('^' + timescale, '', regex=True)
        df = df.reindex([k for k in keys if k in df.index])
        df = df[sorted(df.columns, reverse=True)]
        return df