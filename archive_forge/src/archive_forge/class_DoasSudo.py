from __future__ import annotations
import abc
import shlex
from .util import (
class DoasSudo(Doas):
    """Become using 'doas' in ansible-test and then after bootstrapping use 'sudo' for other ansible commands."""

    @classmethod
    def name(cls) -> str:
        """The name of this plugin."""
        return 'doas_sudo'

    @property
    def method(self) -> str:
        """The name of the Ansible become plugin that is equivalent to this."""
        return 'sudo'