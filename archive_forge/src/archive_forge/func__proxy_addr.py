from base64 import b64encode
import six
from errno import EOPNOTSUPP, EINVAL, EAGAIN
import functools
from io import BytesIO
import logging
import os
from os import SEEK_CUR
import socket
import struct
import sys
def _proxy_addr(self):
    """
        Return proxy address to connect to as tuple object
        """
    proxy_type, proxy_addr, proxy_port, rdns, username, password = self.proxy
    proxy_port = proxy_port or DEFAULT_PORTS.get(proxy_type)
    if not proxy_port:
        raise GeneralProxyError('Invalid proxy type')
    return (proxy_addr, proxy_port)