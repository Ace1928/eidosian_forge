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
def xpandas_read_excel(filepath_or_buffer, download_config: Optional[DownloadConfig]=None, **kwargs):
    import pandas as pd
    if hasattr(filepath_or_buffer, 'read'):
        try:
            return pd.read_excel(filepath_or_buffer, **kwargs)
        except ValueError:
            return pd.read_excel(BytesIO(filepath_or_buffer.read()), **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        try:
            return pd.read_excel(xopen(filepath_or_buffer, 'rb', download_config=download_config), **kwargs)
        except ValueError:
            return pd.read_excel(BytesIO(xopen(filepath_or_buffer, 'rb', download_config=download_config).read()), **kwargs)