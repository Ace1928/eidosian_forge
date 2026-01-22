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
class NetworkRemoteConfig(RemoteConfig, NetworkConfig):
    """Configuration for a remote network host."""
    collection: t.Optional[str] = None
    connection: t.Optional[str] = None

    def get_defaults(self, context: HostContext) -> NetworkRemoteCompletionConfig:
        """Return the default settings."""
        return filter_completion(network_completion()).get(self.name) or NetworkRemoteCompletionConfig(name=self.name, placeholder=True)

    def apply_defaults(self, context: HostContext, defaults: CompletionConfig) -> None:
        """Apply default settings."""
        assert isinstance(defaults, NetworkRemoteCompletionConfig)
        super().apply_defaults(context, defaults)
        self.collection = self.collection or defaults.collection
        self.connection = self.connection or defaults.connection