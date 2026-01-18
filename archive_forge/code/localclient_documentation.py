import atexit
from hashlib import md5
import mimetypes
import os
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from ..client import Client
from ..enums import FileCacheMode
from .localpath import LocalPath
Abstract client for accessing objects the local filesystem. Subclasses are as a monkeypatch
    substitutes for normal Client subclasses when writing tests.