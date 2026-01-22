import collections.abc
import configparser
import enum
import getpass
import json
import logging
import multiprocessing
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from functools import reduce
from typing import (
from urllib.parse import quote, unquote, urlencode, urlparse, urlsplit
from google.protobuf.wrappers_pb2 import BoolValue, DoubleValue, Int32Value, StringValue
import wandb
import wandb.env
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import UsageError
from wandb.proto import wandb_settings_pb2
from wandb.sdk.internal.system.env_probe_helpers import is_aws_lambda
from wandb.sdk.lib import filesystem
from wandb.sdk.lib._settings_toposort_generated import SETTINGS_TOPOLOGICALLY_SORTED
from wandb.sdk.wandb_setup import _EarlyLogger
from .lib import apikey
from .lib.gitlib import GitRepo
from .lib.ipython import _get_python_type
from .lib.runid import generate_id
def _get_wandb_dir(root_dir: str) -> str:
    """Get the full path to the wandb directory.

    The setting exposed to users as `dir=` or `WANDB_DIR` is the `root_dir`.
    We add the `__stage_dir__` to it to get the full `wandb_dir`
    """
    if os.path.exists(os.path.join(root_dir, '.wandb')):
        __stage_dir__ = '.wandb' + os.sep
    else:
        __stage_dir__ = 'wandb' + os.sep
    path = os.path.join(root_dir, __stage_dir__)
    if not os.access(root_dir or '.', os.W_OK):
        wandb.termwarn(f"Path {path} wasn't writable, using system temp directory.", repeat=False)
        path = os.path.join(tempfile.gettempdir(), __stage_dir__ or 'wandb' + os.sep)
    return os.path.expanduser(path)