import errno
import os
import socket
import sys
import six
from ._exceptions import *
from ._logging import *
from ._socket import*
from ._ssl_compat import *
from ._url import *
def _get_addrinfo_list(hostname, port, is_secure, proxy):
    phost, pport, pauth = get_proxy_info(hostname, is_secure, proxy.host, proxy.port, proxy.auth, proxy.no_proxy)
    try:
        if not phost:
            addrinfo_list = socket.getaddrinfo(hostname, port, 0, 0, socket.SOL_TCP)
            return (addrinfo_list, False, None)
        else:
            pport = pport and pport or 80
            addrinfo_list = socket.getaddrinfo(phost, pport, 0, socket.SOCK_STREAM, socket.SOL_TCP)
            return (addrinfo_list, True, pauth)
    except socket.gaierror as e:
        raise WebSocketAddressException(e)