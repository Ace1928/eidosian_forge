from __future__ import annotations
from OpenSSL import SSL
from twisted.internet import ssl
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath

        Return an L{SSL.Context} to be use for server-side connections.

        Will not return a cached context.
        This is done to improve the test coverage as most implementation
        are caching the context.
        