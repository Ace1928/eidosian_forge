from __future__ import annotations
import collections.abc as c
import contextlib
import json
import random
import time
import uuid
import threading
import typing as t
from .util import (
from .util_common import (
from .config import (
from .docker_util import (
from .ansible_util import (
from .core_ci import (
from .target import (
from .ssh import (
from .host_configs import (
from .connections import (
from .thread import (
class ContainerDatabase:
    """Database of running containers used to support tests."""

    def __init__(self, data: dict[str, dict[str, dict[str, ContainerAccess]]]) -> None:
        self.data = data

    @staticmethod
    def from_dict(data: dict[str, t.Any]) -> ContainerDatabase:
        """Return a ContainerDatabase instance from the given dict."""
        return ContainerDatabase(dict(((access_name, dict(((context_name, dict(((container_name, ContainerAccess.from_dict(container)) for container_name, container in containers.items()))) for context_name, containers in contexts.items()))) for access_name, contexts in data.items())))

    def to_dict(self) -> dict[str, t.Any]:
        """Return a dict of the current instance."""
        return dict(((access_name, dict(((context_name, dict(((container_name, container.to_dict()) for container_name, container in containers.items()))) for context_name, containers in contexts.items()))) for access_name, contexts in self.data.items()))