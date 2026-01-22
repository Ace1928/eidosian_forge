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
def _reconfigure_output_stream(self) -> None:
    if self._set_line_buffering is None and self._set_errors is None:
        return
    if not isinstance(self._out, io.TextIOWrapper):
        return
    if hasattr(self._out, 'reconfigure'):
        self._out.reconfigure(line_buffering=self._set_line_buffering, errors=self._set_errors.value if self._set_errors is not None else None)
    elif self._out.line_buffering != self._line_buffering() or self._out.errors != self._errors():
        if hasattr(self._out, '_line_buffering') and hasattr(self._out, '_errors'):
            py_out = typing.cast(Any, self._out)
            py_out._line_buffering = self._line_buffering()
            py_out._errors = self._errors()
            py_out.flush()
        else:
            encoding = self._encoding()
            errors = self._errors()
            line_buffering = self._line_buffering()
            try:
                if self._use_stdout:
                    sys.stdout = typing.cast(TextIO, None)
                newstream = io.TextIOWrapper(self._out.detach(), line_buffering=line_buffering, encoding=encoding, errors=errors)
                self._out = newstream
            finally:
                if self._use_stdout:
                    sys.stdout = self._out