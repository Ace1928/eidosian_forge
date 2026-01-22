from __future__ import annotations
import io
import os
import pathlib
import pytest
from fsspec.utils import (
class CustomFSPath:
    """For testing fspath on unknown objects"""

    def __init__(self, path):
        self.path = path

    def __fspath__(self):
        return self.path