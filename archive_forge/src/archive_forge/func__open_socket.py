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
def _open_socket(addrinfo_list, sockopt, timeout):
    err = None
    for addrinfo in addrinfo_list:
        family, socktype, proto = addrinfo[:3]
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        for opts in DEFAULT_SOCKET_OPTION:
            sock.setsockopt(*opts)
        for opts in sockopt:
            sock.setsockopt(*opts)
        address = addrinfo[4]
        err = None
        while not err:
            try:
                sock.connect(address)
            except ProxyConnectionError as error:
                err = WebSocketProxyException(str(error))
                err.remote_ip = str(address[0])
                continue
            except socket.error as error:
                error.remote_ip = str(address[0])
                try:
                    eConnRefused = (errno.ECONNREFUSED, errno.WSAECONNREFUSED)
                except:
                    eConnRefused = (errno.ECONNREFUSED,)
                if error.errno == errno.EINTR:
                    continue
                elif error.errno in eConnRefused:
                    err = error
                    continue
                else:
                    raise error
            else:
                break
        else:
            continue
        break
    else:
        if err:
            raise err
    return sock