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
def xet_parse(source, parser=None, download_config: Optional[DownloadConfig]=None):
    """Extend `xml.etree.ElementTree.parse` function to support remote files.

    Args:
        source: File path or file object.
        parser (`XMLParser`, *optional*, default `XMLParser`): Parser instance.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `xml.etree.ElementTree.Element`: Root element of the given source document.
    """
    if hasattr(source, 'read'):
        return ET.parse(source, parser=parser)
    else:
        with xopen(source, 'rb', download_config=download_config) as f:
            return ET.parse(f, parser=parser)