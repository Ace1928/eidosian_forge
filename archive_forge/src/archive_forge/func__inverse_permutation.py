import logging
from gzip import GzipFile
from os import PathLike, makedirs, remove
from os.path import exists, join
import joblib
import numpy as np
import scipy.sparse as sp
from ..utils import Bunch
from ..utils import shuffle as shuffle_
from ..utils._param_validation import StrOptions, validate_params
from . import get_data_home
from ._base import RemoteFileMetadata, _fetch_remote, _pkl_filepath, load_descr
from ._svmlight_format_io import load_svmlight_files
def _inverse_permutation(p):
    """Inverse permutation p."""
    n = p.size
    s = np.zeros(n, dtype=np.int32)
    i = np.arange(n, dtype=np.int32)
    np.put(s, p, i)
    return s