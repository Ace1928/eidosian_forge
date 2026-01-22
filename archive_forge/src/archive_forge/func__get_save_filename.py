from __future__ import annotations
import logging # isort:skip
from os.path import abspath, expanduser
from typing import Sequence
from jinja2 import Template
from ..core.templates import FILE
from ..core.types import PathLike
from ..models.ui import UIElement
from ..resources import Resources, ResourcesLike
from ..settings import settings
from ..themes import Theme
from ..util.warnings import warn
from .state import State, curstate
from .util import default_filename
def _get_save_filename(state: State, filename: PathLike | None) -> tuple[PathLike, bool]:
    if filename is not None:
        return (filename, False)
    if state.file and (not settings.ignore_filename()):
        return (state.file.filename, False)
    return (default_filename('html'), True)