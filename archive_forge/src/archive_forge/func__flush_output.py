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
def _flush_output(self) -> None:
    try:
        if not self._out.closed:
            self._out.flush()
    except BrokenPipeError:
        self._exit_code = _signal_exit_code(signal.SIGPIPE)
        try:
            self._out.close()
        except BrokenPipeError:
            pass