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
class PosixSshProfile(SshTargetHostProfile[PosixSshConfig], PosixProfile[PosixSshConfig]):
    """Host profile for a POSIX SSH instance."""

    def get_controller_target_connections(self) -> list[SshConnection]:
        """Return SSH connection(s) for accessing the host as a target from the controller."""
        settings = SshConnectionDetail(name='target', user=self.config.user, host=self.config.host, port=self.config.port, identity_file=SshKey(self.args).key, python_interpreter=self.python.path)
        return [SshConnection(self.args, settings)]