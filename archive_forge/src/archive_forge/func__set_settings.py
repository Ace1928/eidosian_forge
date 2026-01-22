import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
def _set_settings(self, settings):
    object.__setattr__(self, '_settings', settings)