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
def _infer_run_settings_from_environment(self, _logger: Optional[_EarlyLogger]=None) -> None:
    """Modify settings based on environment (for runs only)."""
    settings: Dict[str, Union[bool, str, None]] = dict()
    program = self.program or _get_program()
    if program is not None:
        repo = GitRepo()
        root = repo.root or os.getcwd()
        program_relpath = self.program_relpath or _get_program_relpath(program, repo.root, _logger=_logger)
        settings['program_relpath'] = program_relpath
        program_abspath = os.path.abspath(os.path.join(root, os.path.relpath(os.getcwd(), root), program))
        if os.path.exists(program_abspath):
            settings['program_abspath'] = program_abspath
    else:
        program = '<python with no main file>'
    settings['program'] = program
    if _logger is not None:
        _logger.info(f'Inferring run settings from compute environment: {_redact_dict(settings)}')
    self.update(settings, source=Source.ENV)