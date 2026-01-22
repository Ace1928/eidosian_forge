import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
def _sanitize_dict(self, config_dict, allow_val_change=None, ignore_keys: Optional[set]=None):
    sanitized = {}
    self._raise_value_error_on_nested_artifact(config_dict)
    for k, v in config_dict.items():
        if ignore_keys and k in ignore_keys:
            continue
        k, v = self._sanitize(k, v, allow_val_change)
        sanitized[k] = v
    return sanitized