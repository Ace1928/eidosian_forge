import glob
import io
import os
import posixpath
import re
import tarfile
import time
import xml.dom.minidom
import zipfile
from asyncio import TimeoutError
from io import BytesIO
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET
import fsspec
from aiohttp.client_exceptions import ClientError
from huggingface_hub.utils import EntryNotFoundError
from packaging import version
from .. import config
from ..filesystems import COMPRESSION_FILESYSTEMS
from ..utils.file_utils import (
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .download_config import DownloadConfig
def xexists(urlpath: str, download_config: Optional[DownloadConfig]=None):
    """Extend `os.path.exists` function to support both local and remote files.

    Args:
        urlpath (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `bool`
    """
    main_hop, *rest_hops = _as_str(urlpath).split('::')
    if is_local_path(main_hop):
        return os.path.exists(main_hop)
    else:
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split('::')
        fs, *_ = fsspec.get_fs_token_paths(urlpath, storage_options=storage_options)
        return fs.exists(main_hop)