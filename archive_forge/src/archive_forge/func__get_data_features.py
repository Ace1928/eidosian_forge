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
def _get_data_features(data_id: int, data_home: Optional[str], n_retries: int=3, delay: float=1.0) -> OpenmlFeaturesType:
    url = _DATA_FEATURES.format(data_id)
    error_message = 'Dataset with data_id {} not found.'.format(data_id)
    json_data = _get_json_content_from_openml_api(url, error_message, data_home=data_home, n_retries=n_retries, delay=delay)
    return json_data['data_features']['feature']