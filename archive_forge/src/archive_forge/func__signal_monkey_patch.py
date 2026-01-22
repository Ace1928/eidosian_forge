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
def _signal_monkey_patch(self, signum, handler):
    """Monkey patch for signal.signal that raises an error if a SIGINT handler is
        registered within the DeferSigint context.
        """
    if threading.current_thread() == threading.main_thread() and signum == signal.SIGINT:
        raise ValueError("Can't set signal handler for SIGINT while SIGINT is being deferred within a DeferSigint context.")
    return self.orig_signal(signum, handler)