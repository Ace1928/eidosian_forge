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
def _open_proxied_socket(url, options, proxy):
    hostname, port, resource, is_secure = parse_url(url)
    if not HAS_PYSOCKS:
        raise WebSocketException('PySocks module not found.')
    ptype = socks.SOCKS5
    rdns = False
    if proxy.type == 'socks4':
        ptype = socks.SOCKS4
    if proxy.type == 'http':
        ptype = socks.HTTP
    if proxy.type[-1] == 'h':
        rdns = True
    sock = socks.create_connection((hostname, port), proxy_type=ptype, proxy_addr=proxy.host, proxy_port=proxy.port, proxy_rdns=rdns, proxy_username=proxy.auth[0] if proxy.auth else None, proxy_password=proxy.auth[1] if proxy.auth else None, timeout=options.timeout, socket_options=DEFAULT_SOCKET_OPTION + options.sockopt)
    if is_secure:
        if HAVE_SSL:
            sock = _ssl_socket(sock, options.sslopt, hostname)
        else:
            raise WebSocketException('SSL not available.')
    return (sock, (hostname, port, resource))