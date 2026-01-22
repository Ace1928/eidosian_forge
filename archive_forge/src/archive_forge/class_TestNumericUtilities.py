import datetime
import math
import unittest
from itertools import product
import numpy as np
import pandas as pd
from holoviews import Dimension, Element
from holoviews.core.util import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PointerXY
class TestNumericUtilities(ComparisonTestCase):

    def test_isfinite_none(self):
        self.assertFalse(isfinite(None))

    def test_isfinite_nan(self):
        self.assertFalse(isfinite(float('NaN')))

    def test_isfinite_inf(self):
        self.assertFalse(isfinite(float('inf')))

    def test_isfinite_float(self):
        self.assertTrue(isfinite(1.2))

    def test_isfinite_float_array_nan(self):
        array = np.array([1.2, 3.0, np.nan])
        self.assertEqual(isfinite(array), np.array([True, True, False]))

    def test_isfinite_float_array_inf(self):
        array = np.array([1.2, 3.0, np.inf])
        self.assertEqual(isfinite(array), np.array([True, True, False]))

    def test_isfinite_datetime(self):
        dt = datetime.datetime(2017, 1, 1)
        self.assertTrue(isfinite(dt))

    def test_isfinite_datetime64(self):
        dt64 = np.datetime64(datetime.datetime(2017, 1, 1))
        self.assertTrue(isfinite(dt64))

    def test_isfinite_datetime64_nat(self):
        dt64 = np.datetime64('NaT')
        self.assertFalse(isfinite(dt64))

    def test_isfinite_timedelta64_nat(self):
        dt64 = np.timedelta64('NaT')
        self.assertFalse(isfinite(dt64))

    def test_isfinite_pandas_timestamp_nat(self):
        dt64 = pd.Timestamp('NaT')
        self.assertFalse(isfinite(dt64))

    def test_isfinite_pandas_period_nat(self):
        dt64 = pd.Period('NaT')
        self.assertFalse(isfinite(dt64))

    def test_isfinite_pandas_period_index(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_period('D')
        self.assertEqual(isfinite(daily), np.array([True, True, True]))

    def test_isfinite_pandas_period_series(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_period('D').to_series()
        self.assertEqual(isfinite(daily), np.array([True, True, True]))

    def test_isfinite_pandas_period_index_nat(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_period('D')
        daily = pd.PeriodIndex(list(daily) + [pd.NaT])
        self.assertEqual(isfinite(daily), np.array([True, True, True, False]))

    def test_isfinite_pandas_period_series_nat(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_period('D')
        daily = pd.Series(list(daily) + [pd.NaT])
        self.assertEqual(isfinite(daily), np.array([True, True, True, False]))

    def test_isfinite_pandas_timestamp_index(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D')
        self.assertEqual(isfinite(daily), np.array([True, True, True]))

    def test_isfinite_pandas_timestamp_series(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_series()
        self.assertEqual(isfinite(daily), np.array([True, True, True]))

    def test_isfinite_pandas_timestamp_index_nat(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D')
        daily = pd.DatetimeIndex(list(daily) + [pd.NaT])
        self.assertEqual(isfinite(daily), np.array([True, True, True, False]))

    def test_isfinite_pandas_timestamp_series_nat(self):
        daily = pd.date_range('2017-1-1', '2017-1-3', freq='D')
        daily = pd.Series(list(daily) + [pd.NaT])
        self.assertEqual(isfinite(daily), np.array([True, True, True, False]))

    def test_isfinite_datetime64_array(self):
        dt64 = np.array([np.datetime64(datetime.datetime(2017, 1, i)) for i in range(1, 4)])
        self.assertEqual(isfinite(dt64), np.array([True, True, True]))

    def test_isfinite_datetime64_array_with_nat(self):
        dts = [np.datetime64(datetime.datetime(2017, 1, i)) for i in range(1, 4)]
        dt64 = np.array(dts + [np.datetime64('NaT')])
        self.assertEqual(isfinite(dt64), np.array([True, True, True, False]))