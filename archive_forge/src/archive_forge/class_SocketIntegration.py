from __future__ import absolute_import
import socket
from sentry_sdk import Hub
from sentry_sdk._types import MYPY
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration
class SocketIntegration(Integration):
    identifier = 'socket'

    @staticmethod
    def setup_once():
        """
        patches two of the most used functions of socket: create_connection and getaddrinfo(dns resolver)
        """
        _patch_create_connection()
        _patch_getaddrinfo()