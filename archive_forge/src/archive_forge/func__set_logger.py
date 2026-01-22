import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
def _set_logger(log_object: Logger) -> None:
    """Configure module logger."""
    global logger
    logger = log_object