import os
import shutil
import sys
from typing import (
import wandb
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.hashutil import md5_file_hex
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.wb_value import WBValue
@classmethod
def _tmp_path(cls: Type['_SavedModel']) -> str:
    assert isinstance(cls._path_extension, str), '_path_extension must be a string'
    tmp_path = os.path.abspath(os.path.join(MEDIA_TMP.name, runid.generate_id()))
    if cls._path_extension != '':
        tmp_path += '.' + cls._path_extension
    return tmp_path