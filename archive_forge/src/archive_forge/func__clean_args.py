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
def _clean_args(self, args):
    if isinstance(args, str):
        return self._quote(args)
    if isinstance(args, list):
        return [self._quote(arg) for arg in args]
    return args