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
def _get_cdn_urls(version: str | None=None, minified: bool=True) -> Urls:
    if version is None:
        docs_cdn = settings.docs_cdn()
        version = docs_cdn if docs_cdn else __version__.split('+')[0]
    base_url = _cdn_base_url()
    container = 'bokeh/dev' if _DEV_PAT.match(version) else 'bokeh/release'

    def mk_filename(comp: str, kind: Kind) -> str:
        return f'{comp}-{version}{('.min' if minified else '')}.{kind}'

    def mk_url(comp: str, kind: Kind) -> str:
        return f'{base_url}/{container}/' + mk_filename(comp, kind)
    result = Urls(urls=lambda components, kind: [mk_url(component, kind) for component in components])
    if len(__version__.split('+')) > 1:
        result.messages.append(RuntimeMessage(type='warn', text=f"Requesting CDN BokehJS version '{version}' from local development version '{__version__}'. This configuration is unsupported and may not work!"))
    if is_full_release(version):
        assert version is not None
        sri_hashes = get_sri_hashes_for_version(version)
        result.hashes = lambda components, kind: {mk_url(component, kind): sri_hashes[mk_filename(component, kind)] for component in components}
    return result