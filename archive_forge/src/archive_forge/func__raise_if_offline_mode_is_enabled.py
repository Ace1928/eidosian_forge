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
def _raise_if_offline_mode_is_enabled(msg: Optional[str]=None):
    """Raise an OfflineModeIsEnabled error (subclass of ConnectionError) if HF_DATASETS_OFFLINE is True."""
    if config.HF_DATASETS_OFFLINE:
        raise OfflineModeIsEnabled('Offline mode is enabled.' if msg is None else 'Offline mode is enabled. ' + str(msg))