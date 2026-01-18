import hashlib
import os
import shutil
import sys
from configparser import InterpolationError
from contextlib import contextmanager
from pathlib import Path
from typing import (
import srsly
import typer
from click import NoSuchOption
from click.parser import split_arg_string
from thinc.api import Config, ConfigValidationError, require_gpu
from thinc.util import gpu_is_available
from typer.main import get_command
from wasabi import Printer, msg
from weasel import app as project_cli
from .. import about
from ..compat import Literal
from ..schemas import validate
from ..util import (
def walk_directory(path: Path, suffix: Optional[str]=None) -> List[Path]:
    """Given a directory and a suffix, recursively find all files matching the suffix.
    Directories or files with names beginning with a . are ignored, but hidden flags on
    filesystems are not checked.
    When provided with a suffix `None`, there is no suffix-based filtering."""
    if not path.is_dir():
        return [path]
    paths = [path]
    locs = []
    seen = set()
    for path in paths:
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.parts[-1].startswith('.'):
            continue
        elif path.is_dir():
            paths.extend(path.iterdir())
        elif suffix is not None and (not path.parts[-1].endswith(suffix)):
            continue
        else:
            locs.append(path)
    locs.sort()
    return locs