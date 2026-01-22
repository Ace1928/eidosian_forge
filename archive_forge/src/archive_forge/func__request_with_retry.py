import copy
import io
import json
import multiprocessing
import os
import posixpath
import re
import shutil
import sys
import time
import urllib
import warnings
from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path
from typing import Optional, TypeVar, Union
from unittest.mock import patch
from urllib.parse import urljoin, urlparse
import fsspec
import huggingface_hub
import requests
from fsspec.core import strip_protocol
from fsspec.utils import can_be_local
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from .. import __version__, config
from ..download.download_config import DownloadConfig
from . import _tqdm, logging
from . import tqdm as hf_tqdm
from ._filelock import FileLock
from .extract import ExtractManager
def _request_with_retry(method: str, url: str, max_retries: int=0, base_wait_time: float=0.5, max_wait_time: float=2, timeout: float=10.0, **params) -> requests.Response:
    """Wrapper around requests to retry in case it fails with a ConnectTimeout, with exponential backoff.

    Note that if the environment variable HF_DATASETS_OFFLINE is set to 1, then a OfflineModeIsEnabled error is raised.

    Args:
        method (str): HTTP method, such as 'GET' or 'HEAD'.
        url (str): The URL of the resource to fetch.
        max_retries (int): Maximum number of retries, defaults to 0 (no retries).
        base_wait_time (float): Duration (in seconds) to wait before retrying the first time. Wait time between
            retries then grows exponentially, capped by max_wait_time.
        max_wait_time (float): Maximum amount of time between two retries, in seconds.
        **params (additional keyword arguments): Params to pass to :obj:`requests.request`.
    """
    _raise_if_offline_mode_is_enabled(f'Tried to reach {url}')
    tries, success = (0, False)
    while not success:
        tries += 1
        try:
            response = requests.request(method=method.upper(), url=url, timeout=timeout, **params)
            success = True
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as err:
            if tries > max_retries:
                raise err
            else:
                logger.info(f'{method} request to {url} timed out, retrying... [{tries / max_retries}]')
                sleep_time = min(max_wait_time, base_wait_time * 2 ** (tries - 1))
                time.sleep(sleep_time)
    return response