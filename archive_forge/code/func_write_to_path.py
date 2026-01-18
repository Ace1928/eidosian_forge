import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def write_to_path(self, path: Optional[str]=None) -> None:
    """Write configuration to a file on disk."""
    if path is None:
        path = self.path
    with GitFile(path, 'wb') as f:
        self.write_to_file(f)