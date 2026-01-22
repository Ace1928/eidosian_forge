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
def _check_wandb_core_version_compatibility(core_version: str) -> None:
    """Checks if the installed wandb-core version is compatible with the wandb version."""
    if parse_version(core_version) < parse_version(wandb._minimum_core_version):
        raise ImportError(f'Requires wandb-core version {wandb._minimum_core_version} or later, but you have {core_version}. Run `pip install --upgrade wandb-core` to upgrade.')