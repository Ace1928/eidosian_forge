import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
def _load_viewer(self, settings: Optional[Settings]=None) -> None:
    if self._settings and self._settings._offline:
        return
    if isinstance(settings, dict):
        settings = wandb_settings.Settings(**settings)
    s = server.Server(settings=settings)
    s.query_with_timeout()
    self._server = s