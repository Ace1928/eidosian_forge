from __future__ import annotations
import os
import select
import shlex
import signal
import subprocess
import sys
from typing import ClassVar, Mapping
import param
from pyviz_comms import JupyterComm
from ..io.callbacks import PeriodicCallback
from ..util import edit_readonly, lazy_load
from .base import Widget
def _forward_terminal_input_to_subprocess(self, *events):
    if self._fd:
        os.write(self._fd, self._terminal.value.encode())