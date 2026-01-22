import socket
import sys
import threading
from contextlib import suppress
from . import Adapter
from .. import errors
from .._compat import IS_ABOVE_OPENSSL10
from ..makefile import StreamReader, StreamWriter
from ..server import HTTPServer
It is unclear why exactly this happens.

            It's reproducible only with openssl>1.0 and stdlib
            :py:mod:`ssl` wrapper.
            In CherryPy it's triggered by Checker plugin, which connects
            to the app listening to the socket port in TLS mode via plain
            HTTP during startup (from the same process).


            Ref: https://github.com/cherrypy/cherrypy/issues/1618
            