import csv
import gzip
import hashlib
import os
import shutil
from collections import namedtuple
from importlib import resources
from numbers import Integral
from os import environ, listdir, makedirs
from os.path import expanduser, isdir, join, splitext
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from ..preprocessing import scale
from ..utils import Bunch, check_pandas_support, check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
def _pkl_filepath(*args, **kwargs):
    """Return filename for Python 3 pickles

    args[-1] is expected to be the ".pkl" filename. For compatibility with
    older scikit-learn versions, a suffix is inserted before the extension.

    _pkl_filepath('/path/to/folder', 'filename.pkl') returns
    '/path/to/folder/filename_py3.pkl'

    """
    py3_suffix = kwargs.get('py3_suffix', '_py3')
    basename, ext = splitext(args[-1])
    basename += py3_suffix
    new_args = args[:-1] + (basename + ext,)
    return join(*new_args)