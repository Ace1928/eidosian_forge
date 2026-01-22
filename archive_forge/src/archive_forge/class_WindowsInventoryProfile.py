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
class WindowsInventoryProfile(SshTargetHostProfile[WindowsInventoryConfig]):
    """Host profile for a Windows inventory."""

    def get_controller_target_connections(self) -> list[SshConnection]:
        """Return SSH connection(s) for accessing the host as a target from the controller."""
        inventory = parse_inventory(self.args, self.config.path)
        hosts = get_hosts(inventory, 'windows')
        identity_file = SshKey(self.args).key
        settings = [SshConnectionDetail(name=name, host=config['ansible_host'], port=22, user=config['ansible_user'], identity_file=identity_file, shell_type='powershell') for name, config in hosts.items()]
        if settings:
            details = '\n'.join((f'{ssh.name} {ssh.user}@{ssh.host}:{ssh.port}' for ssh in settings))
            display.info(f'Generated SSH connection details from inventory:\n{details}', verbosity=1)
        return [SshConnection(self.args, setting) for setting in settings]