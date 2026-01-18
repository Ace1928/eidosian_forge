import mimetypes
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from ..client import Client, register_client_class
from ..cloudpath import implementation_registry
from ..enums import FileCacheMode
from ..exceptions import CloudPathException
from .s3path import S3Path
Boto3 query used for quick checks of existence and if path is file/dir