import enum
import io
import os
import posixpath
import tarfile
import warnings
import zipfile
from datetime import datetime
from functools import partial
from itertools import chain
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
from .. import config
from ..utils import tqdm as hf_tqdm
from ..utils.deprecation_utils import DeprecatedEnum, deprecated
from ..utils.file_utils import (
from ..utils.info_utils import get_size_checksum_dict
from ..utils.logging import get_logger
from ..utils.py_utils import NestedDataStructure, map_nested, size_str
from ..utils.track import TrackedIterable, tracked_str
from .download_config import DownloadConfig
class GenerateMode(DeprecatedEnum):
    REUSE_DATASET_IF_EXISTS = 'reuse_dataset_if_exists'
    REUSE_CACHE_IF_EXISTS = 'reuse_cache_if_exists'
    FORCE_REDOWNLOAD = 'force_redownload'

    @property
    def help_message(self):
        return "Use 'DownloadMode' instead."