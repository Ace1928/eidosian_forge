from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _convert_dataframe_1d(idx, silent=False):
    if _check_idx_1d(idx, silent=silent):
        idx = idx.iloc[:, 0] if idx.shape[1] == 1 else idx.iloc[0, :]
    return idx