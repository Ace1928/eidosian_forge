from __future__ import absolute_import
import socket
from sentry_sdk import Hub
from sentry_sdk._types import MYPY
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration
def _patch_getaddrinfo():
    real_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        hub = Hub.current
        if hub.get_integration(SocketIntegration) is None:
            return real_getaddrinfo(host, port, family, type, proto, flags)
        with hub.start_span(op=OP.SOCKET_DNS, description=_get_span_description(host, port)) as span:
            span.set_data('host', host)
            span.set_data('port', port)
            return real_getaddrinfo(host, port, family, type, proto, flags)
    socket.getaddrinfo = getaddrinfo