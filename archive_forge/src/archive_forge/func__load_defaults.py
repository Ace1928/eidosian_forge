import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
def _load_defaults(self):
    conf_dict = config_util.dict_from_config_file('config-defaults.yaml')
    if conf_dict is not None:
        self.update(conf_dict)