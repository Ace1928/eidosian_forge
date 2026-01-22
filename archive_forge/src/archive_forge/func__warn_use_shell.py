from __future__ import annotations
import contextlib
import io
import itertools
import logging
import os
import re
import signal
import subprocess
from subprocess import DEVNULL, PIPE, Popen
import sys
from textwrap import dedent
import threading
import warnings
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import Literal, PathLike, TBD
def _warn_use_shell(extra_danger: bool) -> None:
    warnings.warn(_USE_SHELL_DANGER_MESSAGE if extra_danger else _USE_SHELL_DEFAULT_MESSAGE, DeprecationWarning, stacklevel=3)