import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
def update_locked(self, d, user=None, _allow_val_change=None):
    """Shallow-update config with `d` and lock config updates on d's keys."""
    num = self._get_user_id(user)
    for k, v in d.items():
        k, v = self._sanitize(k, v, allow_val_change=_allow_val_change)
        self._locked[k] = num
        self._items[k] = v
    if self._callback:
        self._callback(data=d)