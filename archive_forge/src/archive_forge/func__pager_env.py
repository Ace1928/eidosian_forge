import contextlib
import enum
import io
import os
import signal
import subprocess
import sys
import types
import typing
from typing import Any, Optional, Type, Dict, TextIO
from autopage import command
def _pager_env(self) -> Optional[Dict[str, str]]:
    new_vars = self._command.environment_variables(self._config)
    if not new_vars:
        return None
    env = dict(os.environ)
    env.update(new_vars)
    return env