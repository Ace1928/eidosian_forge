import logging
import os
import sys
from typing import Optional
import wandb
from ..lib import tracelog
from . import _startup_debug, port_file
from .server_sock import SocketServer
from .streams import StreamMux
def _start_sock(self, mux: StreamMux) -> int:
    address: str = self._address or '127.0.0.1'
    port: int = self._sock_port or 0
    self._sock_server = SocketServer(mux=mux, address=address, port=port)
    try:
        self._sock_server.start()
        port = self._sock_server.port
        if self._pid:
            mux.set_pid(self._pid)
    except KeyboardInterrupt:
        mux.cleanup()
        raise
    except Exception:
        mux.cleanup()
        raise
    return port