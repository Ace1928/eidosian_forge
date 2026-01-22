import typing
import urllib.parse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import List, Optional, Union
from requests import get
from pennylane.data.base import Dataset
from pennylane.data.base.hdf5 import open_hdf5_s3
from .foldermap import DataPath, FolderMapView, ParamArg
from .params import DEFAULT, FULL, format_params
@lru_cache(maxsize=1)
def _get_data_struct():
    """Fetch the data struct from S3."""
    response = get(DATA_STRUCT_URL, timeout=5.0)
    response.raise_for_status()
    return response.json()