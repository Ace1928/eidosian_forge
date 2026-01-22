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
class SupportContainer:
    """Information about a running support container available for use by tests."""

    def __init__(self, container: DockerInspect, container_ip: str, published_ports: dict[int, int]) -> None:
        self.container = container
        self.container_ip = container_ip
        self.published_ports = published_ports