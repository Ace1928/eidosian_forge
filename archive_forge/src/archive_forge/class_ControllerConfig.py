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
class ControllerConfig(PosixConfig):
    """Configuration for the controller host."""
    controller: t.Optional[PosixConfig] = None

    def get_defaults(self, context: HostContext) -> PosixCompletionConfig:
        """Return the default settings."""
        return context.controller_config.get_defaults(context)

    def apply_defaults(self, context: HostContext, defaults: CompletionConfig) -> None:
        """Apply default settings."""
        assert isinstance(defaults, PosixCompletionConfig)
        self.controller = context.controller_config
        if not self.python and (not defaults.supported_pythons):
            self.python = context.controller_config.python
        super().apply_defaults(context, defaults)

    @property
    def is_managed(self) -> bool:
        """
        True if the host is a managed instance, otherwise False.
        Managed instances are used exclusively by ansible-test and can safely have destructive operations performed without explicit permission from the user.
        """
        return self.controller.is_managed

    @property
    def have_root(self) -> bool:
        """True if root is available, otherwise False."""
        return self.controller.have_root