from __future__ import annotations
import logging  # isort:skip
import json
import os
import re
from os.path import relpath
from pathlib import Path
from typing import (
from . import __version__
from .core.templates import CSS_RESOURCES, JS_RESOURCES
from .core.types import ID, PathLike
from .model import Model
from .settings import LogLevel, settings
from .util.dataclasses import dataclass, field
from .util.paths import ROOT_DIR
from .util.token import generate_session_id
from .util.version import is_full_release
def _compute_single_hash(path: Path) -> str:
    assert path.suffix == '.js'
    from subprocess import PIPE, Popen
    digest = f'openssl dgst -sha384 -binary {path}'.split()
    p1 = Popen(digest, stdout=PIPE)
    b64 = 'openssl base64 -A'.split()
    p2 = Popen(b64, stdin=p1.stdout, stdout=PIPE)
    out, _ = p2.communicate()
    return out.decode('utf-8').strip()