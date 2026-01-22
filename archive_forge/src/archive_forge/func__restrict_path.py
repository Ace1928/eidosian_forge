import os
import warnings
from typing import Optional, Sequence, Mapping, Callable
from ipywidgets import Dropdown, Text, Select, Button, HTML
from ipywidgets import Layout, GridBox, Box, HBox, VBox, ValueWidget
from .errors import ParentPathError, InvalidFileNameError
from .utils import get_subpaths, get_dir_contents, match_item, strip_parent_path
from .utils import is_valid_filename, get_drive_letters, normalize_path, has_parent_path
def _restrict_path(self, path) -> str:
    """Calculate the sandboxed path using the sandbox path."""
    if self._sandbox_path == os.sep:
        pass
    elif self._sandbox_path == path:
        path = os.sep
    elif self._sandbox_path:
        if os.path.splitdrive(self._sandbox_path)[0] and len(self._sandbox_path) == 3:
            path = strip_parent_path(path, os.path.splitdrive(self._sandbox_path)[0])
        else:
            path = strip_parent_path(path, self._sandbox_path)
    return path