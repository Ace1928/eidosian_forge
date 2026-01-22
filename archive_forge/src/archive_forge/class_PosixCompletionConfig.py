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
class PosixCompletionConfig(CompletionConfig, metaclass=abc.ABCMeta):
    """Base class for completion configuration of POSIX environments."""

    @property
    @abc.abstractmethod
    def supported_pythons(self) -> list[str]:
        """Return a list of the supported Python versions."""

    @abc.abstractmethod
    def get_python_path(self, version: str) -> str:
        """Return the path of the requested Python version."""

    def get_default_python(self, controller: bool) -> str:
        """Return the default Python version for a controller or target as specified."""
        context_pythons = CONTROLLER_PYTHON_VERSIONS if controller else SUPPORTED_PYTHON_VERSIONS
        version = [python for python in self.supported_pythons if python in context_pythons][0]
        return version

    @property
    def controller_supported(self) -> bool:
        """True if at least one Python version is provided which supports the controller, otherwise False."""
        return any((version in CONTROLLER_PYTHON_VERSIONS for version in self.supported_pythons))