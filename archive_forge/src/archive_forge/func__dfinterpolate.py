import pandas as pd
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
def _dfinterpolate(df, key, column):
    """
    Given a DataFrame, with all values and the Index as floats,
    and given a float key, find the row that matches the key, or 
    find the two rows surrounding that key, and return the interpolated
    value for the specified column, based on where the key falls between
    the two rows.  If they key is an exact match for a key in the index,
    the return the exact value from the column.  If the key is less than
    or greater than any key in the index, then return either the first
    or last value for the column.
    """
    s = df[column]
    s1 = s.loc[:key]
    if len(s1) < 1:
        return s.iloc[0]
    j1 = s1.index[-1]
    v1 = s1.iloc[-1]
    s2 = s.loc[key:]
    if len(s2) < 1:
        return s.iloc[-1]
    j2 = s2.index[0]
    v2 = s2.iloc[0]
    if j1 == j2:
        return v1
    delta = j2 - j1
    portion = (key - j1) / delta
    ans = v1 + (v2 - v1) * portion
    return ans