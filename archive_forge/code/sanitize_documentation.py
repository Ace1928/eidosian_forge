from . import utils
import numpy as np
import pandas as pd
import warnings
Ensure that the data index is unique in a pandas DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    copy : bool, optional (default: True)
        If True, return a modified copy of the data. Otherwise modify it in place.

    Returns
    -------
    data : pd.DataFrame
        Sanitized data
    