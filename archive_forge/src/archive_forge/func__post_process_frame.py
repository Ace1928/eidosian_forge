import itertools
import re
from collections import OrderedDict
from collections.abc import Generator
from typing import List
import numpy as np
import scipy as sp
from ..externals import _arff
from ..externals._arff import ArffSparseDataType
from ..utils import (
from ..utils.fixes import pd_fillna
def _post_process_frame(frame, feature_names, target_names):
    """Post process a dataframe to select the desired columns in `X` and `y`.

    Parameters
    ----------
    frame : dataframe
        The dataframe to split into `X` and `y`.

    feature_names : list of str
        The list of feature names to populate `X`.

    target_names : list of str
        The list of target names to populate `y`.

    Returns
    -------
    X : dataframe
        The dataframe containing the features.

    y : {series, dataframe} or None
        The series or dataframe containing the target.
    """
    X = frame[feature_names]
    if len(target_names) >= 2:
        y = frame[target_names]
    elif len(target_names) == 1:
        y = frame[target_names[0]]
    else:
        y = None
    return (X, y)