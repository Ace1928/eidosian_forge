import socket
from .base import StatsClientBase, PipelineBase
def _do_send(self, data):
    self._sock.sendall(data.encode('ascii') + b'\n')