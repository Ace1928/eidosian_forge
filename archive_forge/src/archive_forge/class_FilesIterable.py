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
class FilesIterable(_IterableFromGenerator):
    """An iterable of paths from a list of directories or files"""

    @classmethod
    def _iter_from_urlpaths(cls, urlpaths: Union[str, List[str]], download_config: Optional[DownloadConfig]=None) -> Generator[str, None, None]:
        if not isinstance(urlpaths, list):
            urlpaths = [urlpaths]
        for urlpath in urlpaths:
            if xisfile(urlpath, download_config=download_config):
                yield urlpath
            elif xisdir(urlpath, download_config=download_config):
                for dirpath, dirnames, filenames in xwalk(urlpath, download_config=download_config):
                    dirnames[:] = sorted([dirname for dirname in dirnames if not dirname.startswith(('.', '__'))])
                    if xbasename(dirpath).startswith(('.', '__')):
                        continue
                    for filename in sorted(filenames):
                        if filename.startswith(('.', '__')):
                            continue
                        yield xjoin(dirpath, filename)
            else:
                raise FileNotFoundError(urlpath)

    @classmethod
    def from_urlpaths(cls, urlpaths, download_config: Optional[DownloadConfig]=None) -> 'FilesIterable':
        return cls(cls._iter_from_urlpaths, urlpaths, download_config)