import abc
from collections import defaultdict
import collections.abc
from contextlib import contextmanager
import os
from pathlib import (  # type: ignore
import shutil
import sys
from typing import (
from urllib.parse import urlparse
from warnings import warn
from cloudpathlib.enums import FileCacheMode
from . import anypath
from .exceptions import (
def _upload_file_to_cloud(self, local_path: Path, force_overwrite_to_cloud: bool=False) -> Self:
    """Uploads file at `local_path` to the cloud if there is not a newer file
        already there.
        """
    try:
        stats = self.stat()
    except NoStatError:
        stats = None
    if not stats or local_path.stat().st_mtime > stats.st_mtime or force_overwrite_to_cloud:
        self.client._upload_file(local_path, self)
        return self
    raise OverwriteNewerCloudError(f'Local file ({self._local}) for cloud path ({self}) is newer in the cloud disk, but is being requested to be uploaded to the cloud. Either (1) redownload changes from the cloud or (2) pass `force_overwrite_to_cloud=True` to overwrite.')