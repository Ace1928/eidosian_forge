from statsmodels.compat.python import asstr, lmap, lrange, lzip
import datetime
import re
import numpy as np
from pandas import to_datetime

    Turns a sequence of date strings and returns a list of datetime.

    Parameters
    ----------
    start : str
        The first abbreviated date, for instance, '1965q1' or '1965m1'
    end : str, optional
        The last abbreviated date if length is None.
    length : int, optional
        The length of the returned array of end is None.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd
    >>> nobs = 50
    >>> dates = pd.date_range('1960m1', length=nobs)


    Returns
    -------
    date_list : ndarray
        A list of datetime types.
    