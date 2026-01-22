import os
import platform
import socket
import ssl
import typing
import _ssl  # type: ignore[import]
from ._ssl_constants import (
class TruststoreSSLObject(ssl.SSLObject):

    def do_handshake(self) -> None:
        ret = super().do_handshake()
        _verify_peercerts(self, server_hostname=self.server_hostname)
        return ret