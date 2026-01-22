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
def _create_connected_socket(self, channel: str, identity: bytes | None=None) -> zmq.sugar.socket.Socket:
    """Create a zmq Socket and connect it to the kernel."""
    url = self._make_url(channel)
    socket_type = channel_socket_types[channel]
    self.log.debug('Connecting to: %s', url)
    sock = self.context.socket(socket_type)
    sock.linger = 1000
    if identity:
        sock.identity = identity
    sock.connect(url)
    return sock