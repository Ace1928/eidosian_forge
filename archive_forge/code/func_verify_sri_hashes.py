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
def verify_sri_hashes() -> None:
    """ Verify the SRI hashes in a full release package.

    This function compares the computed SRI hashes for the BokehJS files in a
    full release package to the values in the SRI manifest file. Returns None
    if all hashes match, otherwise an exception will be raised.

    .. note::
        This function can only be called on full release (e.g "1.2.3") packages.

    Returns:
        None

    Raises:
        ValueError
            If called outside a full release package
        RuntimeError
            If there are missing, extra, or mismatched files

    """
    if not is_full_release():
        raise ValueError('verify_sri_hashes() can only be used with full releases')
    paths = list((settings.bokehjs_path() / 'js').glob('bokeh*.js'))
    hashes = get_sri_hashes_for_version(__version__)
    if len(hashes) < len(paths):
        raise RuntimeError("There are unexpected 'bokeh*.js' files in the package")
    if len(hashes) > len(paths):
        raise RuntimeError("There are 'bokeh*.js' files missing in the package")
    bad: list[Path] = []
    for path in paths:
        name, suffix = str(path.name).split('.', 1)
        filename = f'{name}-{__version__}.{suffix}'
        sri_hash = _compute_single_hash(path)
        if hashes[filename] != sri_hash:
            bad.append(path)
    if bad:
        raise RuntimeError(f'SRI Hash mismatches in the package: {bad!r}')