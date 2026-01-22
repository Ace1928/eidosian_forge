import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
def _update_user_settings(self, settings: Optional[Settings]=None) -> None:
    settings = settings or self._settings
    self._server = None
    user_settings = self._load_user_settings(settings=settings)
    if user_settings is not None:
        self._settings._apply_user(user_settings)