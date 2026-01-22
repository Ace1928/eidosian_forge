from __future__ import annotations
import asyncio
import csv
import io
import json
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from jupyter_server.base.handlers import APIHandler
from tornado import web
from traitlets import List, Unicode
from traitlets.config import LoggingConfigurable
from .config import get_federated_extensions
def app_static_info(self) -> tuple[Path | None, str | None]:
    """get the static directory for this app

        This will usually be in `static_dir`, but may also appear in the
        parent of `static_dir`.
        """
    if TYPE_CHECKING:
        from .app import LabServerApp
        assert isinstance(self.parent, LabServerApp)
    path = Path(self.parent.static_dir)
    package_json = path / 'package.json'
    if not package_json.exists():
        parent_package_json = path.parent / 'package.json'
        if parent_package_json.exists():
            package_json = parent_package_json
        else:
            return (None, None)
    name = json.loads(package_json.read_text(encoding='utf-8'))['name']
    return (path, name)