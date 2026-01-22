import hashlib
import os
import platform
import re
import shutil
from typing import TYPE_CHECKING, Optional, Sequence, Type, Union, cast
import wandb
from wandb import util
from wandb._globals import _datatypes_callback
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.paths import LogicalPath
from .wb_value import WBValue
def _set_file(self, path: str, is_tmp: bool=False, extension: Optional[str]=None) -> None:
    self._path = path
    self._is_tmp = is_tmp
    self._extension = extension
    assert extension is None or path.endswith(extension), f'Media file extension "{extension}" must occur at the end of path "{path}".'
    with open(self._path, 'rb') as f:
        self._sha256 = hashlib.sha256(f.read()).hexdigest()
    self._size = os.path.getsize(self._path)