from __future__ import annotations
import abc
import collections.abc as c
import enum
import fcntl
import importlib.util
import inspect
import json
import keyword
import os
import platform
import pkgutil
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import functools
import shlex
import typing as t
import warnings
from struct import unpack, pack
from termios import TIOCGWINSZ
from .locale_util import (
from .encoding import (
from .io import (
from .thread import (
from .constants import (
class HostConnectionError(ApplicationError):
    """
    Raised when the initial connection during host profile setup has failed and all retries have been exhausted.
    Raised by provisioning code when one or more provisioning threads raise this exception.
    Also raised when an SSH connection fails for the shell command.
    """

    def __init__(self, message: str, callback: t.Callable[[], None]=None) -> None:
        super().__init__(message)
        self._callback = callback

    def run_callback(self) -> None:
        """Run the error callback, if any."""
        if self._callback:
            self._callback()