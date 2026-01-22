import enum
import io
import multiprocessing
import os
import posixpath
import warnings
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Union
import fsspec
from fsspec.core import url_to_fs
from tqdm.contrib.concurrent import thread_map
from .. import config
from ..utils import tqdm as hf_tqdm
from ..utils.deprecation_utils import DeprecatedEnum, deprecated
from ..utils.file_utils import (
from ..utils.info_utils import get_size_checksum_dict
from ..utils.logging import get_logger, tqdm
from ..utils.py_utils import NestedDataStructure, map_nested, size_str
from ..utils.track import tracked_str
from .download_config import DownloadConfig
def _download_batched(self, url_or_filenames: List[str], download_config: DownloadConfig) -> List[str]:
    if len(url_or_filenames) >= 16:
        download_config = download_config.copy()
        download_config.disable_tqdm = True
        download_func = partial(self._download_single, download_config=download_config)
        fs: fsspec.AbstractFileSystem
        fs, path = url_to_fs(url_or_filenames[0], **download_config.storage_options)
        size = 0
        try:
            size = fs.info(path).get('size', 0)
        except Exception:
            pass
        max_workers = config.HF_DATASETS_MULTITHREADING_MAX_WORKERS if size < 20 << 20 else 1
        return thread_map(download_func, url_or_filenames, desc=download_config.download_desc or 'Downloading', unit='files', position=multiprocessing.current_process()._identity[-1] if os.environ.get('HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS') == '1' and multiprocessing.current_process()._identity else None, max_workers=max_workers, tqdm_class=tqdm)
    else:
        return [self._download_single(url_or_filename, download_config=download_config) for url_or_filename in url_or_filenames]