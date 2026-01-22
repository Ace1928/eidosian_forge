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
def _find_all_matching_keys(d: Dict, match_fn: Callable[[Any], bool], visited: Optional[Set[int]]=None, key_path: Tuple[Any, ...]=()) -> Generator[Tuple[Tuple[Any, ...], Any], None, None]:
    """Recursively find all keys that satisfies a match function.

    Args:
       d: The dict to search.
       match_fn: The function to determine if the key is a match.
       visited: Keep track of visited nodes so we dont recurse forever.
       key_path: Keep track of all the keys to get to the current node.

    Yields:
       (key_path, key): The location where the key was found, and the key
    """
    if visited is None:
        visited = set()
    me = id(d)
    if me not in visited:
        visited.add(me)
        for key, value in d.items():
            if match_fn(key):
                yield (key_path, key)
            if isinstance(value, dict):
                yield from _find_all_matching_keys(value, match_fn, visited=visited, key_path=tuple(list(key_path) + [key]))