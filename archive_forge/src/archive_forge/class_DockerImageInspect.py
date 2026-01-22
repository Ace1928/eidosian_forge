from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import (
from .util_common import (
from .config import (
from .thread import (
from .cgroup import (
class DockerImageInspect:
    """The results of `docker image inspect` for a single image."""

    def __init__(self, args: CommonConfig, inspection: dict[str, t.Any]) -> None:
        self.args = args
        self.inspection = inspection

    @property
    def config(self) -> dict[str, t.Any]:
        """Return a dictionary of the image config."""
        return self.inspection['Config']

    @property
    def volumes(self) -> dict[str, t.Any]:
        """Return a dictionary of the image volumes."""
        return self.config.get('Volumes') or {}

    @property
    def cmd(self) -> list[str]:
        """The command to run when the container starts."""
        return self.config['Cmd']