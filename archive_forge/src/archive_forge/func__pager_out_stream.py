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
def _pager_out_stream(self) -> Optional[TextIO]:
    if not self._use_stdout:
        try:
            self._out.fileno()
        except OSError:
            pass
        else:
            return self._out
    return None