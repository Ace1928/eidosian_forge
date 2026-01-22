from __future__ import annotations
import typing as t
from jupyter_client.manager import KernelManager
from nbclient.client import NotebookClient
from nbclient.client import execute as _execute
from nbclient.exceptions import CellExecutionError  # noqa: F401
from nbformat import NotebookNode
from .base import Preprocessor
def _check_assign_resources(self, resources):
    if resources or not hasattr(self, 'resources'):
        self.resources = resources