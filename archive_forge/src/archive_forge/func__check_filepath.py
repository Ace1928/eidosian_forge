import os
import string
import sys
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf  # type: ignore
from tensorflow.keras import callbacks  # type: ignore
import wandb
from wandb.sdk.lib import telemetry
from wandb.sdk.lib.paths import StrPath
from ..keras import patch_tf_keras
def _check_filepath(self) -> None:
    placeholders = []
    for tup in string.Formatter().parse(self.filepath):
        if tup[1] is not None:
            placeholders.append(tup[1])
    if len(placeholders) == 0:
        wandb.termwarn('When using `save_best_only`, ensure that the `filepath` argument contains formatting placeholders like `{epoch:02d}` or `{batch:02d}`. This ensures correct interpretation of the logged artifacts.', repeat=False)