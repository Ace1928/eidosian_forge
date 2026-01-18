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
def xwalk(urlpath, download_config: Optional[DownloadConfig]=None, **kwargs):
    """Extend `os.walk` function to support remote files.

    Args:
        urlpath (`str`): URL root path.
        download_config : mainly use token or storage_options to support different platforms and auth types.
        **kwargs: Additional keyword arguments forwarded to the underlying filesystem.


    Yields:
        `tuple`: 3-tuple (dirpath, dirnames, filenames).
    """
    main_hop, *rest_hops = _as_str(urlpath).split('::')
    if is_local_path(main_hop):
        yield from os.walk(main_hop, **kwargs)
    else:
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split('::')
        fs, *_ = fsspec.get_fs_token_paths(urlpath, storage_options=storage_options)
        inner_path = main_hop.split('://')[-1]
        if inner_path.strip('/') and (not fs.isdir(inner_path)):
            return []
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[-1]
        for dirpath, dirnames, filenames in fs.walk(inner_path, **kwargs):
            yield ('::'.join([f'{protocol}://{dirpath}'] + rest_hops), dirnames, filenames)