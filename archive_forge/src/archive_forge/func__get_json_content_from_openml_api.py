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
def _get_json_content_from_openml_api(url: str, error_message: Optional[str], data_home: Optional[str], n_retries: int=3, delay: float=1.0) -> Dict:
    """
    Loads json data from the openml api.

    Parameters
    ----------
    url : str
        The URL to load from. Should be an official OpenML endpoint.

    error_message : str or None
        The error message to raise if an acceptable OpenML error is thrown
        (acceptable error is, e.g., data id not found. Other errors, like 404's
        will throw the native error message).

    data_home : str or None
        Location to cache the response. None if no cache is required.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    json_data : json
        the json result from the OpenML server if the call was successful.
        An exception otherwise.
    """

    @_retry_with_clean_cache(url, data_home=data_home)
    def _load_json():
        with closing(_open_openml_url(url, data_home, n_retries=n_retries, delay=delay)) as response:
            return json.loads(response.read().decode('utf-8'))
    try:
        return _load_json()
    except HTTPError as error:
        if error.code != 412:
            raise error
    raise OpenMLError(error_message)