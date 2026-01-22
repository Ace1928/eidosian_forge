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
class RemoteProfile(SshTargetHostProfile[TRemoteConfig], metaclass=abc.ABCMeta):
    """Base class for remote instance profiles."""

    @property
    def core_ci_state(self) -> t.Optional[dict[str, str]]:
        """The saved Ansible Core CI state."""
        return self.state.get('core_ci')

    @core_ci_state.setter
    def core_ci_state(self, value: dict[str, str]) -> None:
        """The saved Ansible Core CI state."""
        self.state['core_ci'] = value

    def provision(self) -> None:
        """Provision the host before delegation."""
        self.core_ci = self.create_core_ci(load=True)
        self.core_ci.start()
        self.core_ci_state = self.core_ci.save()

    def deprovision(self) -> None:
        """Deprovision the host after delegation has completed."""
        if self.args.remote_terminate == TerminateMode.ALWAYS or (self.args.remote_terminate == TerminateMode.SUCCESS and self.args.success):
            self.delete_instance()

    @property
    def core_ci(self) -> t.Optional[AnsibleCoreCI]:
        """Return the cached AnsibleCoreCI instance, if any, otherwise None."""
        return self.cache.get('core_ci')

    @core_ci.setter
    def core_ci(self, value: AnsibleCoreCI) -> None:
        """Cache the given AnsibleCoreCI instance."""
        self.cache['core_ci'] = value

    def get_instance(self) -> t.Optional[AnsibleCoreCI]:
        """Return the current AnsibleCoreCI instance, loading it if not already loaded."""
        if not self.core_ci and self.core_ci_state:
            self.core_ci = self.create_core_ci(load=False)
            self.core_ci.load(self.core_ci_state)
        return self.core_ci

    def delete_instance(self) -> None:
        """Delete the AnsibleCoreCI VM instance."""
        core_ci = self.get_instance()
        if not core_ci:
            return
        core_ci.stop()

    def wait_for_instance(self) -> AnsibleCoreCI:
        """Wait for an AnsibleCoreCI VM instance to become ready."""
        core_ci = self.get_instance()
        core_ci.wait()
        return core_ci

    def create_core_ci(self, load: bool) -> AnsibleCoreCI:
        """Create and return an AnsibleCoreCI instance."""
        if not self.config.arch:
            raise InternalError(f'No arch specified for config: {self.config}')
        return AnsibleCoreCI(args=self.args, resource=VmResource(platform=self.config.platform, version=self.config.version, architecture=self.config.arch, provider=self.config.provider, tag='controller' if self.controller else 'target'), load=load)