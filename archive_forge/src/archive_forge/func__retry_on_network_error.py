import gzip
import hashlib
import json
import os
import shutil
import time
from contextlib import closing
from functools import wraps
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from warnings import warn
import numpy as np
from ..utils import (
from ..utils._param_validation import (
from . import get_data_home
from ._arff_parser import load_arff_from_gzip_file
def _retry_on_network_error(n_retries: int=3, delay: float=1.0, url: str='') -> Callable:
    """If the function call results in a network error, call the function again
    up to ``n_retries`` times with a ``delay`` between each call. If the error
    has a 412 status code, don't call the function again as this is a specific
    OpenML error.
    The url parameter is used to give more information to the user about the
    error.
    """

    def decorator(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            retry_counter = n_retries
            while True:
                try:
                    return f(*args, **kwargs)
                except (URLError, TimeoutError) as e:
                    if isinstance(e, HTTPError) and e.code == 412:
                        raise
                    if retry_counter == 0:
                        raise
                    warn(f'A network error occurred while downloading {url}. Retrying...')
                    retry_counter -= 1
                    time.sleep(delay)
        return wrapper
    return decorator