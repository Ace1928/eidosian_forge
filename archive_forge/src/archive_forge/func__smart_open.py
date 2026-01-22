from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
from typing import Any
from jupyter_core.application import JupyterApp
from traitlets import Bool, Unicode
from ._version import __version__
from .config import LabConfig
from .workspaces_handler import WorkspacesManager
def _smart_open(self) -> Any:
    file_name = self.extra_args[0]
    if file_name == '-':
        return sys.stdin
    file_path = Path(file_name).resolve()
    if not file_path.exists():
        self.log.info('%s does not exist.', file_name)
        self.exit(1)
    return file_path.open(encoding='utf-8')