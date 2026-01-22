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
def _apply_config_files(self, _logger: Optional[_EarlyLogger]=None) -> None:
    if self.settings_system is not None:
        if _logger is not None:
            _logger.info(f'Loading settings from {self.settings_system}')
        self.update(self._load_config_file(self.settings_system), source=Source.SYSTEM)
    if self.settings_workspace is not None:
        if _logger is not None:
            _logger.info(f'Loading settings from {self.settings_workspace}')
        self.update(self._load_config_file(self.settings_workspace), source=Source.WORKSPACE)