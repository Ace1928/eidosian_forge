from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
def update_environment(self, **kwargs: Any) -> Dict[str, Union[str, None]]:
    """Set environment variables for future git invocations. Return all changed
        values in a format that can be passed back into this function to revert the
        changes.

        ``Examples``::

            old_env = self.update_environment(PWD='/tmp')
            self.update_environment(**old_env)

        :param kwargs: Environment variables to use for git processes

        :return: Dict that maps environment variables to their old values
        """
    old_env = {}
    for key, value in kwargs.items():
        if value is not None:
            old_env[key] = self._environment.get(key)
            self._environment[key] = value
        elif key in self._environment:
            old_env[key] = self._environment[key]
            del self._environment[key]
    return old_env