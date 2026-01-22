from __future__ import annotations
import abc
import dataclasses
import os
import shlex
import tempfile
import time
import typing as t
from .io import (
from .config import (
from .host_configs import (
from .core_ci import (
from .util import (
from .util_common import (
from .docker_util import (
from .bootstrap import (
from .venv import (
from .ssh import (
from .ansible_util import (
from .containers import (
from .connections import (
from .become import (
from .completion import (
from .dev.container_probe import (
class HostProfile(t.Generic[THostConfig], metaclass=abc.ABCMeta):
    """Base class for host profiles."""

    def __init__(self, *, args: EnvironmentConfig, config: THostConfig, targets: t.Optional[list[HostConfig]]) -> None:
        self.args = args
        self.config = config
        self.controller = bool(targets)
        self.targets = targets or []
        self.state: dict[str, t.Any] = {}
        'State that must be persisted across delegation.'
        self.cache: dict[str, t.Any] = {}
        'Cache that must not be persisted across delegation.'

    def provision(self) -> None:
        """Provision the host before delegation."""

    def setup(self) -> None:
        """Perform out-of-band setup before delegation."""

    def on_target_failure(self) -> None:
        """Executed during failure handling if this profile is a target."""

    def deprovision(self) -> None:
        """Deprovision the host after delegation has completed."""

    def wait(self) -> None:
        """Wait for the instance to be ready. Executed before delegation for the controller and after delegation for targets."""

    def configure(self) -> None:
        """Perform in-band configuration. Executed before delegation for the controller and after delegation for targets."""

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key not in ('args', 'cache')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cache = {}