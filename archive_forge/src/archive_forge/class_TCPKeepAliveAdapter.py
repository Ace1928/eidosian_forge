import socket
import warnings
import sys
import requests
from requests import adapters
from .._compat import connection
from .._compat import poolmanager
from .. import exceptions as exc
class TCPKeepAliveAdapter(SocketOptionsAdapter):
    """An adapter for requests that turns on TCP Keep-Alive by default.

    The adapter sets 4 socket options:

    - ``SOL_SOCKET`` ``SO_KEEPALIVE`` - This turns on TCP Keep-Alive
    - ``IPPROTO_TCP`` ``TCP_KEEPINTVL`` 20 - Sets the keep alive interval
    - ``IPPROTO_TCP`` ``TCP_KEEPCNT`` 5 - Sets the number of keep alive probes
    - ``IPPROTO_TCP`` ``TCP_KEEPIDLE`` 60 - Sets the keep alive time if the
      socket library has the ``TCP_KEEPIDLE`` constant

    The latter three can be overridden by keyword arguments (respectively):

    - ``interval``
    - ``count``
    - ``idle``

    You can use this adapter like so::

       >>> from requests_toolbelt.adapters import socket_options
       >>> tcp = socket_options.TCPKeepAliveAdapter(idle=120, interval=10)
       >>> s = requests.Session()
       >>> s.mount('http://', tcp)

    """

    def __init__(self, **kwargs):
        socket_options = kwargs.pop('socket_options', SocketOptionsAdapter.default_options)
        idle = kwargs.pop('idle', 60)
        interval = kwargs.pop('interval', 20)
        count = kwargs.pop('count', 5)
        socket_options = socket_options + [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        if getattr(socket, 'TCP_KEEPINTVL', None) is not None:
            socket_options += [(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval)]
        elif sys.platform == 'darwin':
            TCP_KEEPALIVE = getattr(socket, 'TCP_KEEPALIVE', 16)
            socket_options += [(socket.IPPROTO_TCP, TCP_KEEPALIVE, interval)]
        if getattr(socket, 'TCP_KEEPCNT', None) is not None:
            socket_options += [(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, count)]
        if getattr(socket, 'TCP_KEEPIDLE', None) is not None:
            socket_options += [(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle)]
        super(TCPKeepAliveAdapter, self).__init__(socket_options=socket_options, **kwargs)