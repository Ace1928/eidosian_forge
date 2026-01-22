import logging
import os
import sys
from typing import Optional
import wandb
from ..lib import tracelog
from . import _startup_debug, port_file
from .server_sock import SocketServer
from .streams import StreamMux
def _stop_servers(self) -> None:
    if self._sock_server:
        self._sock_server.stop()