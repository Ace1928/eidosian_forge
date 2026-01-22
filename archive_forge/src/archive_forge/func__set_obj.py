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
def _set_obj(self, model_obj: Any) -> None:
    assert model_obj is not None and self._validate_obj(model_obj), f'Invalid model object {model_obj}'
    self._model_obj = model_obj