import io
import logging
from pathlib import Path
from typing import IO, Any, Dict, Union
import fsspec
import fsspec.utils
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem
from lightning_utilities.core.imports import module_available
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
def _is_object_storage(fs: AbstractFileSystem) -> bool:
    if module_available('adlfs'):
        from adlfs import AzureBlobFileSystem
        if isinstance(fs, AzureBlobFileSystem):
            return True
    if module_available('gcsfs'):
        from gcsfs import GCSFileSystem
        if isinstance(fs, GCSFileSystem):
            return True
    if module_available('s3fs'):
        from s3fs import S3FileSystem
        if isinstance(fs, S3FileSystem):
            return True
    return False