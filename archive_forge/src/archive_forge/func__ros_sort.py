import warnings
import numpy as np
import pandas as pd
from scipy import stats
def _ros_sort(df, observations, censorship, warn=False):
    """
    This function prepares a dataframe for ROS.

    It sorts ascending with
    left-censored observations first. Censored observations larger than
    the maximum uncensored observations are removed from the dataframe.

    Parameters
    ----------
    df : DataFrame

    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    ------
    sorted_df : DataFrame
        The sorted dataframe with all columns dropped except the
        observation and censorship columns.
    """
    censored = df[df[censorship]].sort_values(observations, axis=0)
    uncensored = df[~df[censorship]].sort_values(observations, axis=0)
    if censored[observations].max() > uncensored[observations].max():
        censored = censored[censored[observations] <= uncensored[observations].max()]
        if warn:
            msg = 'Dropping censored observations greater than the max uncensored observation.'
            warnings.warn(msg)
    combined = pd.concat([censored, uncensored], axis=0)
    return combined[[observations, censorship]].reset_index(drop=True)