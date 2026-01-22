from __future__ import (absolute_import, division,
from future.builtins import bytes, int, str, super
from future.utils import PY2
from future.backports.email import parser as email_parser
from future.backports.email import message as email_message
from future.backports.misc import create_connection as socket_create_connection
import io
import os
import socket
from future.backports.urllib.parse import urlsplit
import warnings
from array import array
def _set_hostport(self, host, port):
    if port is None:
        i = host.rfind(':')
        j = host.rfind(']')
        if i > j:
            try:
                port = int(host[i + 1:])
            except ValueError:
                if host[i + 1:] == '':
                    port = self.default_port
                else:
                    raise InvalidURL("nonnumeric port: '%s'" % host[i + 1:])
            host = host[:i]
        else:
            port = self.default_port
        if host and host[0] == '[' and (host[-1] == ']'):
            host = host[1:-1]
    self.host = host
    self.port = port