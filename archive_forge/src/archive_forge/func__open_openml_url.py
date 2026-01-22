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
def _open_openml_url(openml_path: str, data_home: Optional[str], n_retries: int=3, delay: float=1.0):
    """
    Returns a resource from OpenML.org. Caches it to data_home if required.

    Parameters
    ----------
    openml_path : str
        OpenML URL that will be accessed. This will be prefixes with
        _OPENML_PREFIX.

    data_home : str
        Directory to which the files will be cached. If None, no caching will
        be applied.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    result : stream
        A stream to the OpenML resource.
    """

    def is_gzip_encoded(_fsrc):
        return _fsrc.info().get('Content-Encoding', '') == 'gzip'
    req = Request(_OPENML_PREFIX + openml_path)
    req.add_header('Accept-encoding', 'gzip')
    if data_home is None:
        fsrc = _retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(req)
        if is_gzip_encoded(fsrc):
            return gzip.GzipFile(fileobj=fsrc, mode='rb')
        return fsrc
    local_path = _get_local_path(openml_path, data_home)
    dir_name, file_name = os.path.split(local_path)
    if not os.path.exists(local_path):
        os.makedirs(dir_name, exist_ok=True)
        try:
            with TemporaryDirectory(dir=dir_name) as tmpdir:
                with closing(_retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(req)) as fsrc:
                    opener: Callable
                    if is_gzip_encoded(fsrc):
                        opener = open
                    else:
                        opener = gzip.GzipFile
                    with opener(os.path.join(tmpdir, file_name), 'wb') as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                shutil.move(fdst.name, local_path)
        except Exception:
            if os.path.exists(local_path):
                os.unlink(local_path)
            raise
    return gzip.GzipFile(local_path, 'rb')