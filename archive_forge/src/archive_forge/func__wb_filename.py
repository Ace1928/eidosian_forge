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
def _wb_filename(key: Union[str, int], step: Union[str, int], id: Union[str, int], extension: str) -> str:
    return f'{str(key)}_{str(step)}_{str(id)}{extension}'