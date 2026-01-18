import colorsys
import contextlib
import dataclasses
import functools
import gzip
import importlib
import importlib.util
import itertools
import json
import logging
import math
import numbers
import os
import platform
import queue
import random
import re
import secrets
import shlex
import socket
import string
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from importlib import import_module
from sys import getsizeof
from types import ModuleType
from typing import (
import requests
import yaml
import wandb
import wandb.env
from wandb.errors import AuthenticationError, CommError, UsageError, term
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.json_util import dump, dumps
from wandb.sdk.lib.paths import FilePathStr, StrPath
def vendor_setup() -> Callable:
    """Create a function that restores user paths after vendor imports.

    This enables us to use the vendor directory for packages we don't depend on. Call
    the returned function after imports are complete. If you don't you may modify the
    user's path which is never good.

    Usage:

    ```python
    reset_path = vendor_setup()
    # do any vendor imports...
    reset_path()
    ```
    """
    original_path = [directory for directory in sys.path]

    def reset_import_path() -> None:
        sys.path = original_path
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    vendor_dir = os.path.join(parent_dir, 'vendor')
    vendor_packages = ('gql-0.2.0', 'graphql-core-1.1', 'watchdog_0_9_0', 'promise-2.3.0')
    package_dirs = [os.path.join(vendor_dir, p) for p in vendor_packages]
    for p in [vendor_dir] + package_dirs:
        if p not in sys.path:
            sys.path.insert(1, p)
    return reset_import_path