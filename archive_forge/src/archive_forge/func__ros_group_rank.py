import warnings
import numpy as np
import pandas as pd
from scipy import stats
def _ros_group_rank(df, dl_idx, censorship):
    """
    Ranks each observation within the data groups.

    In this case, the groups are defined by the record's detection
    limit index and censorship status.

    Parameters
    ----------
    df : DataFrame

    dl_idx : str
        Name of the column in the dataframe the index of the
        observations' corresponding detection limit in the `cohn`
        dataframe.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    -------
    ranks : ndarray
        Array of ranks for the dataset.
    """
    ranks = df.copy()
    ranks.loc[:, 'rank'] = 1
    ranks = ranks.groupby(by=[dl_idx, censorship])['rank'].transform(lambda g: g.cumsum())
    return ranks