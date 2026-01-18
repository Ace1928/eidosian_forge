from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_regex_replace_list_mixed(self, mix_ab):
    dfmix = DataFrame(mix_ab)
    to_replace_res = ['\\s*\\.\\s*', 'a']
    values = [np.nan, 'crap']
    mix2 = {'a': list(range(4)), 'b': list('ab..'), 'c': list('halo')}
    dfmix2 = DataFrame(mix2)
    res = dfmix2.replace(to_replace_res, values, regex=True)
    expec = DataFrame({'a': mix2['a'], 'b': ['crap', 'b', np.nan, np.nan], 'c': ['h', 'crap', 'l', 'o']})
    tm.assert_frame_equal(res, expec)
    to_replace_res = ['\\s*(\\.)\\s*', '(a|b)']
    values = ['\\1\\1', '\\1_crap']
    res = dfmix.replace(to_replace_res, values, regex=True)
    expec = DataFrame({'a': mix_ab['a'], 'b': ['a_crap', 'b_crap', '..', '..']})
    tm.assert_frame_equal(res, expec)
    to_replace_res = ['\\s*(\\.)\\s*', 'a', '(b)']
    values = ['\\1\\1', 'crap', '\\1_crap']
    res = dfmix.replace(to_replace_res, values, regex=True)
    expec = DataFrame({'a': mix_ab['a'], 'b': ['crap', 'b_crap', '..', '..']})
    tm.assert_frame_equal(res, expec)
    to_replace_res = ['\\s*(\\.)\\s*', 'a', '(b)']
    values = ['\\1\\1', 'crap', '\\1_crap']
    res = dfmix.replace(regex=to_replace_res, value=values)
    expec = DataFrame({'a': mix_ab['a'], 'b': ['crap', 'b_crap', '..', '..']})
    tm.assert_frame_equal(res, expec)