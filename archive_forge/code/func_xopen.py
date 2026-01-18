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
def xopen(file: str, mode='r', *args, download_config: Optional[DownloadConfig]=None, **kwargs):
    """Extend `open` function to support remote files using `fsspec`.

    It also has a retry mechanism in case connection fails.
    The `args` and `kwargs` are passed to `fsspec.open`, except `token` which is used for queries to private repos on huggingface.co

    Args:
        file (`str`): Path name of the file to be opened.
        mode (`str`, *optional*, default "r"): Mode in which the file is opened.
        *args: Arguments to be passed to `fsspec.open`.
        download_config : mainly use token or storage_options to support different platforms and auth types.
        **kwargs: Keyword arguments to be passed to `fsspec.open`.

    Returns:
        file object
    """
    file_str = _as_str(file)
    main_hop, *rest_hops = file_str.split('::')
    if is_local_path(main_hop):
        kwargs.pop('block_size', None)
        return open(main_hop, mode, *args, **kwargs)
    file, storage_options = _prepare_path_and_storage_options(file_str, download_config=download_config)
    kwargs = {**kwargs, **(storage_options or {})}
    try:
        file_obj = fsspec.open(file, *args, mode=mode, **kwargs).open()
    except ValueError as e:
        if str(e) == 'Cannot seek streaming HTTP file':
            raise NonStreamableDatasetError("Streaming is not possible for this dataset because data host server doesn't support HTTP range requests. You can still load this dataset in non-streaming mode by passing `streaming=False` (default)") from e
        else:
            raise
    except FileNotFoundError:
        if file.startswith(config.HF_ENDPOINT):
            raise FileNotFoundError(file + '\nIf the repo is private or gated, make sure to log in with `huggingface-cli login`.') from None
        else:
            raise
    _add_retries_to_file_obj_read_method(file_obj)
    return file_obj