import abc
import collections
import collections.abc
import os
import sys
import typing
from typing import Optional, Dict, List
class CustomPager(PagerCommand):
    """A pager command parsed from a user-specified string."""

    def __init__(self, pager_cmdline: str):
        import shlex
        self._cmd = shlex.split(pager_cmdline)

    def command(self) -> List[str]:
        return self._cmd

    def environment_variables(self, config: PagerConfig) -> Optional[Dict[str, str]]:
        env = {}
        for cmd_provider in (Less, LV):
            cmd_env = cmd_provider().environment_variables(config)
            if cmd_env:
                env.update(cmd_env)
        return env or None