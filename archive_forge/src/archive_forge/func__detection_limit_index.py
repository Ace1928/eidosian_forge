import warnings
import numpy as np
import pandas as pd
from scipy import stats
def _detection_limit_index(obs, cohn):
    """
    Locates the corresponding detection limit for each observation.

    Basically, creates an array of indices for the detection limits
    (Cohn numbers) corresponding to each data point.

    Parameters
    ----------
    obs : float
        A single observation from the larger dataset.

    cohn : DataFrame
        DataFrame of Cohn numbers.

    Returns
    -------
    det_limit_index : int
        The index of the corresponding detection limit in `cohn`

    See Also
    --------
    cohn_numbers
    """
    if cohn.shape[0] > 0:
        index, = np.where(cohn['lower_dl'] <= obs)
        det_limit_index = index[-1]
    else:
        det_limit_index = 0
    return det_limit_index