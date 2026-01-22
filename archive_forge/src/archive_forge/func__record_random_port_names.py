from __future__ import annotations
import errno
import glob
import json
import os
import socket
import stat
import tempfile
import warnings
from getpass import getpass
from typing import TYPE_CHECKING, Any, Dict, Union, cast
import zmq
from jupyter_core.paths import jupyter_data_dir, jupyter_runtime_dir, secure_write
from traitlets import Bool, CaselessStrEnum, Instance, Integer, Type, Unicode, observe
from traitlets.config import LoggingConfigurable, SingletonConfigurable
from .localinterfaces import localhost
from .utils import _filefind
def _record_random_port_names(self) -> None:
    """Records which of the ports are randomly assigned.

        Records on first invocation, if the transport is tcp.
        Does nothing on later invocations."""
    if self.transport != 'tcp':
        return
    if self._random_port_names is not None:
        return
    self._random_port_names = []
    for name in port_names:
        if getattr(self, name) <= 0:
            self._random_port_names.append(name)