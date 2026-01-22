import queue
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from wandb.proto import wandb_server_pb2 as spb
from wandb.sdk.internal.settings_static import SettingsStatic
from ..lib import tracelog
from ..lib.sock_client import SockClient, SockClientClosedError
from .streams import StreamMux
class SockAcceptThread(threading.Thread):
    _sock: socket.socket
    _mux: StreamMux
    _stopped: 'Event'
    _clients: ClientDict

    def __init__(self, sock: socket.socket, mux: StreamMux) -> None:
        self._sock = sock
        self._mux = mux
        self._stopped = mux._get_stopped_event()
        threading.Thread.__init__(self)
        self.name = 'SockAcceptThr'
        self._clients = ClientDict()

    def run(self) -> None:
        self._sock.listen(5)
        read_threads = []
        while not self._stopped.is_set():
            try:
                conn, addr = self._sock.accept()
            except ConnectionAbortedError:
                break
            except OSError:
                break
            sr = SockServerReadThread(conn=conn, mux=self._mux, clients=self._clients)
            sr.start()
            read_threads.append(sr)
        for rt in read_threads:
            rt.stop()