from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
class TargetMode(enum.Enum):
    """Type of provisioning to use for the targets."""
    WINDOWS_INTEGRATION = enum.auto()
    NETWORK_INTEGRATION = enum.auto()
    POSIX_INTEGRATION = enum.auto()
    SANITY = enum.auto()
    UNITS = enum.auto()
    SHELL = enum.auto()
    NO_TARGETS = enum.auto()

    @property
    def one_host(self) -> bool:
        """Return True if only one host (the controller) should be used, otherwise return False."""
        return self in (TargetMode.SANITY, TargetMode.UNITS, TargetMode.NO_TARGETS)

    @property
    def no_fallback(self) -> bool:
        """Return True if no fallback is acceptable for the controller (due to options not applying to the target), otherwise return False."""
        return self in (TargetMode.WINDOWS_INTEGRATION, TargetMode.NETWORK_INTEGRATION, TargetMode.NO_TARGETS)

    @property
    def multiple_pythons(self) -> bool:
        """Return True if multiple Python versions are allowed, otherwise False."""
        return self in (TargetMode.SANITY, TargetMode.UNITS)

    @property
    def has_python(self) -> bool:
        """Return True if this mode uses Python, otherwise False."""
        return self in (TargetMode.POSIX_INTEGRATION, TargetMode.SANITY, TargetMode.UNITS, TargetMode.SHELL)