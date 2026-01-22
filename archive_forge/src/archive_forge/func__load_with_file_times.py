from __future__ import annotations
import hashlib
import json
import re
import unicodedata
import urllib
from pathlib import Path
from typing import Any
from jupyter_server import _tz as tz
from jupyter_server.base.handlers import APIHandler
from jupyter_server.extension.handler import ExtensionHandlerJinjaMixin, ExtensionHandlerMixin
from jupyter_server.utils import url_path_join as ujoin
from tornado import web
from traitlets.config import LoggingConfigurable
def _load_with_file_times(workspace_path: Path) -> dict:
    """
    Load workspace JSON from disk, overwriting the `created` and `last_modified`
    metadata with current file stat information
    """
    stat = workspace_path.stat()
    with workspace_path.open(encoding='utf-8') as fid:
        workspace = json.load(fid)
        workspace['metadata'].update(last_modified=tz.utcfromtimestamp(stat.st_mtime).isoformat(), created=tz.utcfromtimestamp(stat.st_ctime).isoformat())
    return workspace