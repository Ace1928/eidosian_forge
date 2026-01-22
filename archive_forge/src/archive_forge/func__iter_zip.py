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
@staticmethod
def _iter_zip(f):
    zipf = zipfile.ZipFile(f)
    for member in zipf.infolist():
        file_path = member.filename
        if member.is_dir():
            continue
        if file_path is None:
            continue
        if os.path.basename(file_path).startswith(('.', '__')):
            continue
        file_obj = zipf.open(member)
        yield (file_path, file_obj)