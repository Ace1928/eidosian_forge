from __future__ import annotations
import fnmatch
import os
import warnings
from pathlib import Path
from typing import Any, Callable, List, Literal
from gradio_client.documentation import document
from gradio.components.base import Component, server
from gradio.data_classes import GradioRootModel
def _strip_root(self, path):
    if path.startswith(self.root_dir):
        return path[len(self.root_dir) + 1:]
    return path