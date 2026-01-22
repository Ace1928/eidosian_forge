import os
import types
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
from filelock import BaseFileLock, Timeout
from . import config
from .arrow_dataset import Dataset
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager
from .features import Features
from .info import DatasetInfo, MetricInfo
from .naming import camelcase_to_snakecase
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger
from .utils.py_utils import copyfunc, temp_seed
def _build_data_dir(self):
    """Path of this metric in cache_dir:
        Will be:
            self._data_dir_root/self.name/self.config_name/self.hash (if not none)/
        If any of these element is missing or if ``with_version=False`` the corresponding subfolders are dropped.
        """
    builder_data_dir = self._data_dir_root
    builder_data_dir = os.path.join(builder_data_dir, self.name, self.config_name)
    os.makedirs(builder_data_dir, exist_ok=True)
    return builder_data_dir