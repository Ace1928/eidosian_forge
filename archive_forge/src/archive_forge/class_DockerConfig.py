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
@dataclasses.dataclass
class DockerConfig(ControllerHostConfig, PosixConfig):
    """Configuration for a docker host."""
    name: t.Optional[str] = None
    image: t.Optional[str] = None
    memory: t.Optional[int] = None
    privileged: t.Optional[bool] = None
    seccomp: t.Optional[str] = None
    cgroup: t.Optional[CGroupVersion] = None
    audit: t.Optional[AuditMode] = None

    def get_defaults(self, context: HostContext) -> DockerCompletionConfig:
        """Return the default settings."""
        return filter_completion(docker_completion()).get(self.name) or DockerCompletionConfig(name=self.name, image=self.name, placeholder=True)

    def get_default_targets(self, context: HostContext) -> list[ControllerConfig]:
        """Return the default targets for this host config."""
        if self.name in filter_completion(docker_completion()):
            defaults = self.get_defaults(context)
            pythons = {version: defaults.get_python_path(version) for version in defaults.supported_pythons}
        else:
            pythons = {context.controller_config.python.version: context.controller_config.python.path}
        return [ControllerConfig(python=NativePythonConfig(version=version, path=path)) for version, path in pythons.items()]

    def apply_defaults(self, context: HostContext, defaults: CompletionConfig) -> None:
        """Apply default settings."""
        assert isinstance(defaults, DockerCompletionConfig)
        super().apply_defaults(context, defaults)
        self.name = defaults.name
        self.image = defaults.image
        if self.seccomp is None:
            self.seccomp = defaults.seccomp
        if self.cgroup is None:
            self.cgroup = defaults.cgroup_enum
        if self.audit is None:
            self.audit = defaults.audit_enum
        if self.privileged is None:
            self.privileged = False

    @property
    def is_managed(self) -> bool:
        """
        True if this host is a managed instance, otherwise False.
        Managed instances are used exclusively by ansible-test and can safely have destructive operations performed without explicit permission from the user.
        """
        return True

    @property
    def have_root(self) -> bool:
        """True if root is available, otherwise False."""
        return True