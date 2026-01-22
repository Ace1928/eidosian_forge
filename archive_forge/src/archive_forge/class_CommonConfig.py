from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
class CommonConfig:
    """Configuration common to all commands."""

    def __init__(self, args: t.Any, command: str) -> None:
        self.command = command
        self.interactive = False
        self.check_layout = True
        self.success: t.Optional[bool] = None
        self.color: bool = args.color
        self.explain: bool = args.explain
        self.verbosity: int = args.verbosity
        self.debug: bool = args.debug
        self.truncate: int = args.truncate
        self.redact: bool = args.redact
        self.display_stderr: bool = False
        self.session_name = generate_name()
        self.cache: dict[str, t.Any] = {}

    def get_ansible_config(self) -> str:
        """Return the path to the Ansible config for the given config."""
        return os.path.join(ANSIBLE_TEST_DATA_ROOT, 'ansible.cfg')