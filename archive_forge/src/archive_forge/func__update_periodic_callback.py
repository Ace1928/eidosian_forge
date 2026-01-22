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
@param.depends('_period', watch=True)
def _update_periodic_callback(self):
    if self._periodic_callback:
        self._periodic_callback.period = self._period