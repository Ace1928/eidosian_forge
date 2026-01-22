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
def _link_and_save_file(path: str, base_path: str, interface: 'InterfaceQueue', settings: 'SettingsStatic') -> None:
    files_dir = settings.files_dir
    file_name = os.path.relpath(path, base_path)
    abs_path = os.path.abspath(path)
    wandb_path = os.path.join(files_dir, file_name)
    filesystem.mkdir_exists_ok(os.path.dirname(wandb_path))
    if os.path.islink(wandb_path) and abs_path != os.readlink(wandb_path):
        os.remove(wandb_path)
        os.symlink(abs_path, wandb_path)
    elif not os.path.exists(wandb_path):
        os.symlink(abs_path, wandb_path)
    interface.publish_files(dict(files=[(GlobStr(glob.escape(file_name)), 'live')]))