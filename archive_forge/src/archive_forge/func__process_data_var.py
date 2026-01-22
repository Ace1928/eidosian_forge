import logging
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from .. import utils
from ..rcparams import rcParams
from .base import CoordSpec, DimSpec, dict_to_dataset, infer_stan_dtypes, requires
from .inference_data import InferenceData
def _process_data_var(string):
    """Transform datastring to key, values pair.

    All values are transformed to floating point values.

    Parameters
    ----------
    string : str

    Returns
    -------
    Tuple[Str, Str]
        key, values pair
    """
    key, var = string.split('<-')
    if 'structure' in var:
        var, dim = var.replace('structure(', '').replace(',', '').split('.Dim')
        dtype = float
        var = var.replace('c(', '').replace(')', '').strip().split()
        dim = dim.replace('=', '').replace('c(', '').replace(')', '').strip().split()
        dim = tuple(map(int, dim))
        var = np.fromiter(map(dtype, var), dtype).reshape(dim, order='F')
    elif 'c(' in var:
        dtype = float
        var = var.replace('c(', '').replace(')', '').split(',')
        var = np.fromiter(map(dtype, var), dtype)
    else:
        dtype = float
        var = dtype(var)
    return (key.strip(), var)