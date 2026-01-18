import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def try_to_symlink(symlink_path, target_path):
    """Attempt to create a symlink.

    If the symlink path exists and isn't a symlink, the symlink will not be
    created. If a symlink exists in the path, it will be attempted to be
    removed and replaced.

    Args:
        symlink_path: The path at which to create the symlink.
        target_path: The path the symlink should point to.
    """
    symlink_path = os.path.expanduser(symlink_path)
    target_path = os.path.expanduser(target_path)
    if os.path.exists(symlink_path):
        if os.path.islink(symlink_path):
            try:
                os.remove(symlink_path)
            except OSError:
                return
        else:
            return
    try:
        os.symlink(target_path, symlink_path)
    except OSError:
        return