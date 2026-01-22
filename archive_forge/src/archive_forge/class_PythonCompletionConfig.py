from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
@dataclasses.dataclass(frozen=True)
class PythonCompletionConfig(PosixCompletionConfig, metaclass=abc.ABCMeta):
    """Base class for completion configuration of Python environments."""
    python: str = ''
    python_dir: str = '/usr/bin'

    @property
    def supported_pythons(self) -> list[str]:
        """Return a list of the supported Python versions."""
        versions = self.python.split(',') if self.python else []
        versions = [version for version in versions if version in SUPPORTED_PYTHON_VERSIONS]
        return versions

    def get_python_path(self, version: str) -> str:
        """Return the path of the requested Python version."""
        return os.path.join(self.python_dir, f'python{version}')