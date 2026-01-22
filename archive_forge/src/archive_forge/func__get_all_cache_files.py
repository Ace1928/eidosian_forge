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
def _get_all_cache_files(self) -> Tuple[List[str], List[FileLock]]:
    """Get a lock on all the cache files in a distributed setup.
        We wait for timeout second to let all the distributed node finish their tasks (default is 100 seconds).
        """
    if self.num_process == 1:
        if self.cache_file_name is None:
            raise ValueError("Metric cache file doesn't exist. Please make sure that you call `add` or `add_batch` at least once before calling `compute`.")
        file_paths = [self.cache_file_name]
    else:
        file_paths = [os.path.join(self.data_dir, f'{self.experiment_id}-{self.num_process}-{process_id}.arrow') for process_id in range(self.num_process)]
    filelocks = []
    for process_id, file_path in enumerate(file_paths):
        if process_id == 0:
            filelocks.append(self.filelock)
        else:
            filelock = FileLock(file_path + '.lock')
            try:
                filelock.acquire(timeout=self.timeout)
            except Timeout:
                raise ValueError(f'Cannot acquire lock on cached file {file_path} for process {process_id}.') from None
            else:
                filelocks.append(filelock)
    return (file_paths, filelocks)