import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
class SafeTransport(Transport):
    """Handles an HTTPS transaction to an XML-RPC server."""

    def __init__(self, use_datetime=False, use_builtin_types=False, *, headers=(), context=None):
        super().__init__(use_datetime=use_datetime, use_builtin_types=use_builtin_types, headers=headers)
        self.context = context

    def make_connection(self, host):
        if self._connection and host == self._connection[0]:
            return self._connection[1]
        if not hasattr(http.client, 'HTTPSConnection'):
            raise NotImplementedError("your version of http.client doesn't support HTTPS")
        chost, self._extra_headers, x509 = self.get_host_info(host)
        self._connection = (host, http.client.HTTPSConnection(chost, None, context=self.context, **x509 or {}))
        return self._connection[1]