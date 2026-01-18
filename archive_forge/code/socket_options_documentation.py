import socket
import warnings
import sys
import requests
from requests import adapters
from .._compat import connection
from .._compat import poolmanager
from .. import exceptions as exc
An adapter for requests that turns on TCP Keep-Alive by default.

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

    