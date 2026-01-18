from __future__ import annotations
from datetime import timedelta
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.fields import isleapyear_arr
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
import pandas.core.common as com
def to_timestamp(self, freq=None, how: str='start') -> DatetimeArray:
    """
        Cast to DatetimeArray/Index.

        Parameters
        ----------
        freq : str or DateOffset, optional
            Target frequency. The default is 'D' for week or longer,
            's' otherwise.
        how : {'s', 'e', 'start', 'end'}
            Whether to use the start or end of the time period being converted.

        Returns
        -------
        DatetimeArray/Index

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.to_timestamp()
        DatetimeIndex(['2023-01-01', '2023-02-01', '2023-03-01'],
        dtype='datetime64[ns]', freq='MS')
        """
    from pandas.core.arrays import DatetimeArray
    how = libperiod.validate_end_alias(how)
    end = how == 'E'
    if end:
        if freq == 'B' or self.freq == 'B':
            adjust = Timedelta(1, 'D') - Timedelta(1, 'ns')
            return self.to_timestamp(how='start') + adjust
        else:
            adjust = Timedelta(1, 'ns')
            return (self + self.freq).to_timestamp(how='start') - adjust
    if freq is None:
        freq_code = self._dtype._get_to_timestamp_base()
        dtype = PeriodDtypeBase(freq_code, 1)
        freq = dtype._freqstr
        base = freq_code
    else:
        freq = Period._maybe_convert_freq(freq)
        base = freq._period_dtype_code
    new_parr = self.asfreq(freq, how=how)
    new_data = libperiod.periodarr_to_dt64arr(new_parr.asi8, base)
    dta = DatetimeArray._from_sequence(new_data)
    if self.freq.name == 'B':
        diffs = libalgos.unique_deltas(self.asi8)
        if len(diffs) == 1:
            diff = diffs[0]
            if diff == self.dtype._n:
                dta._freq = self.freq
            elif diff == 1:
                dta._freq = self.freq.base
        return dta
    else:
        return dta._with_freq('infer')