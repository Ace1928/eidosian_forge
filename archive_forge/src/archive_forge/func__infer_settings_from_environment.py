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
def _infer_settings_from_environment(self, _logger: Optional[_EarlyLogger]=None) -> None:
    """Modify settings based on environment (for runs and cli)."""
    settings: Dict[str, Union[bool, str, Sequence, None]] = dict()
    settings['symlink'] = True
    if self._windows:
        settings['symlink'] = False
    if (self.save_code is True or self.save_code is None) and (os.getenv(wandb.env.SAVE_CODE) is not None or os.getenv(wandb.env.DISABLE_CODE) is not None):
        settings['save_code'] = wandb.env.should_save_code()
    settings['disable_git'] = wandb.env.disable_git()
    if self._jupyter and (self.notebook_name is None or self.notebook_name == ''):
        meta = wandb.jupyter.notebook_metadata(self.silent)
        settings['_jupyter_path'] = meta.get('path')
        settings['_jupyter_name'] = meta.get('name')
        settings['_jupyter_root'] = meta.get('root')
    elif self._jupyter and self.notebook_name is not None and os.path.exists(self.notebook_name):
        settings['_jupyter_path'] = self.notebook_name
        settings['_jupyter_name'] = self.notebook_name
        settings['_jupyter_root'] = os.getcwd()
    elif self._jupyter:
        wandb.termwarn(f"WANDB_NOTEBOOK_NAME should be a path to a notebook file, couldn't find {self.notebook_name}.")
    if self.host is None:
        settings['host'] = socket.gethostname()
    if self.username is None:
        try:
            settings['username'] = getpass.getuser()
        except KeyError:
            settings['username'] = str(os.getuid())
    _executable = self._executable or os.environ.get(wandb.env._EXECUTABLE) or sys.executable or shutil.which('python3') or 'python3'
    settings['_executable'] = _executable
    settings['docker'] = wandb.env.get_docker(wandb.util.image_id_from_k8s())
    if os.path.exists('/usr/local/cuda/version.txt'):
        with open('/usr/local/cuda/version.txt') as f:
            settings['_cuda'] = f.read().split(' ')[-1].strip()
    if not self._jupyter:
        settings['_args'] = sys.argv[1:]
    settings['_os'] = platform.platform(aliased=True)
    settings['_python'] = platform.python_version()
    if self._windows and self._except_exit is None:
        settings['_except_exit'] = True
    if _logger is not None:
        _logger.info(f'Inferring settings from compute environment: {_redact_dict(settings)}')
    self.update(settings, source=Source.ENV)