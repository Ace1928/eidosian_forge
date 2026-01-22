from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class DtypeWarning(Warning):
    """
    Warning raised when reading different dtypes in a column from a file.

    Raised for a dtype incompatibility. This can happen whenever `read_csv`
    or `read_table` encounter non-uniform dtypes in a column(s) of a given
    CSV file.

    See Also
    --------
    read_csv : Read CSV (comma-separated) file into a DataFrame.
    read_table : Read general delimited file into a DataFrame.

    Notes
    -----
    This warning is issued when dealing with larger files because the dtype
    checking happens per chunk read.

    Despite the warning, the CSV file is read with mixed types in a single
    column which will be an object type. See the examples below to better
    understand this issue.

    Examples
    --------
    This example creates and reads a large CSV file with a column that contains
    `int` and `str`.

    >>> df = pd.DataFrame({'a': (['1'] * 100000 + ['X'] * 100000 +
    ...                          ['1'] * 100000),
    ...                    'b': ['b'] * 300000})  # doctest: +SKIP
    >>> df.to_csv('test.csv', index=False)  # doctest: +SKIP
    >>> df2 = pd.read_csv('test.csv')  # doctest: +SKIP
    ... # DtypeWarning: Columns (0) have mixed types

    Important to notice that ``df2`` will contain both `str` and `int` for the
    same input, '1'.

    >>> df2.iloc[262140, 0]  # doctest: +SKIP
    '1'
    >>> type(df2.iloc[262140, 0])  # doctest: +SKIP
    <class 'str'>
    >>> df2.iloc[262150, 0]  # doctest: +SKIP
    1
    >>> type(df2.iloc[262150, 0])  # doctest: +SKIP
    <class 'int'>

    One way to solve this issue is using the `dtype` parameter in the
    `read_csv` and `read_table` functions to explicit the conversion:

    >>> df2 = pd.read_csv('test.csv', sep=',', dtype={'a': str})  # doctest: +SKIP

    No warning was issued.
    """