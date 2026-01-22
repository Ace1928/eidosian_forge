from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
import re
import sys
from os.path import (
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Sequence
from ..core.has_props import HasProps
from ..settings import settings
from .strings import snakify
def _npmjs_path() -> Path:
    global _npmjs
    if _npmjs is None:
        executable = 'npm.cmd' if sys.platform == 'win32' else 'npm'
        _npmjs = _nodejs_path().parent / executable
    return _npmjs