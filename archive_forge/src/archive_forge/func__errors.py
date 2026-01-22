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
def _errors(self) -> str:
    if self._set_errors is None:
        return getattr(self._out, 'errors', ErrorStrategy.STRICT.value)
    return self._set_errors.value