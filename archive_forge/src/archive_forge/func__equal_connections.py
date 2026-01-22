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
@staticmethod
def _equal_connections(conn1: KernelConnectionInfo, conn2: KernelConnectionInfo) -> bool:
    """Compares pertinent keys of connection info data. Returns True if equivalent, False otherwise."""
    pertinent_keys = ['key', 'ip', 'stdin_port', 'iopub_port', 'shell_port', 'control_port', 'hb_port', 'transport', 'signature_scheme']
    return all((conn1.get(key) == conn2.get(key) for key in pertinent_keys))