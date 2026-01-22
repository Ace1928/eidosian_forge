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
@staticmethod
def _validate__async_upload_concurrency_limit(value: int) -> bool:
    if value <= 0:
        raise UsageError('_async_upload_concurrency_limit must be positive')
    try:
        import resource
        file_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception:
        pass
    else:
        if value > file_limit:
            wandb.termwarn(f"_async_upload_concurrency_limit setting of {value} exceeds this process's limit on open files ({file_limit}); may cause file-upload failures. Try decreasing _async_upload_concurrency_limit, or increasing your file limit with `ulimit -n`.", repeat=False)
    return True