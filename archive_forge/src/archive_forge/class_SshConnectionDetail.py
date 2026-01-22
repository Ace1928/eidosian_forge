from __future__ import annotations
import dataclasses
import itertools
import json
import os
import random
import re
import subprocess
import shlex
import typing as t
from .encoding import (
from .util import (
from .config import (
@dataclasses.dataclass
class SshConnectionDetail:
    """Information needed to establish an SSH connection to a host."""
    name: str
    host: str
    port: t.Optional[int]
    user: str
    identity_file: str
    python_interpreter: t.Optional[str] = None
    shell_type: t.Optional[str] = None
    enable_rsa_sha1: bool = False

    def __post_init__(self):
        self.name = sanitize_host_name(self.name)

    @property
    def options(self) -> dict[str, str]:
        """OpenSSH config options, which can be passed to the `ssh` CLI with the `-o` argument."""
        options: dict[str, str] = {}
        if self.enable_rsa_sha1:
            algorithms = '+ssh-rsa'
            options.update(HostKeyAlgorithms=algorithms, PubkeyAcceptedKeyTypes=algorithms)
        return options