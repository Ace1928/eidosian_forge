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
class HostContext:
    """Context used when getting and applying defaults for host configurations."""
    controller_config: t.Optional['PosixConfig']

    @property
    def controller(self) -> bool:
        """True if the context is for the controller, otherwise False."""
        return not self.controller_config