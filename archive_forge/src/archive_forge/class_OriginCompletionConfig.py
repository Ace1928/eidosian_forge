from __future__ import annotations
import abc
import dataclasses
import enum
import os
import pickle
import sys
import typing as t
from .constants import (
from .io import (
from .completion import (
from .util import (
@dataclasses.dataclass(frozen=True)
class OriginCompletionConfig(PosixCompletionConfig):
    """Pseudo completion config for the origin."""

    def __init__(self) -> None:
        super().__init__(name='origin')

    @property
    def supported_pythons(self) -> list[str]:
        """Return a list of the supported Python versions."""
        current_version = version_to_str(sys.version_info[:2])
        versions = [version for version in SUPPORTED_PYTHON_VERSIONS if version == current_version] + [version for version in SUPPORTED_PYTHON_VERSIONS if version != current_version]
        return versions

    def get_python_path(self, version: str) -> str:
        """Return the path of the requested Python version."""
        version = find_python(version)
        return version

    @property
    def is_default(self) -> bool:
        """True if the completion entry is only used for defaults, otherwise False."""
        return False