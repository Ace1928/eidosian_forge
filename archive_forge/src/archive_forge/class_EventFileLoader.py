import glob
import logging
import os
import queue
import socket
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import wandb
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib import filesystem
from wandb.viz import CustomChart
from . import run as internal_run
class EventFileLoader(event_file_loader.EventFileLoader):

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        if save:
            if REMOTE_FILE_TOKEN in file_path:
                logger.warning('Not persisting remote tfevent file: %s', file_path)
            else:
                logdir = os.path.dirname(file_path)
                parts = list(os.path.split(logdir))
                if namespace and parts[-1] == namespace:
                    parts.pop()
                    logdir = os.path.join(*parts)
                _link_and_save_file(path=file_path, base_path=logdir, interface=_loader_interface, settings=_loader_settings)