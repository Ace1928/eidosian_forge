from __future__ import absolute_import
import io
import logging
import zlib
from contextlib import contextmanager
from socket import error as SocketError
from socket import timeout as SocketTimeout
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPException
from .exceptions import (
from .packages import six
from .util.response import is_fp_closed, is_response_to_head
@contextmanager
def _error_catcher(self):
    """
        Catch low-level python exceptions, instead re-raising urllib3
        variants, so that low-level exceptions are not leaked in the
        high-level api.

        On exit, release the connection back to the pool.
        """
    clean_exit = False
    try:
        try:
            yield
        except SocketTimeout:
            raise ReadTimeoutError(self._pool, None, 'Read timed out.')
        except BaseSSLError as e:
            if 'read operation timed out' not in str(e):
                raise SSLError(e)
            raise ReadTimeoutError(self._pool, None, 'Read timed out.')
        except (HTTPException, SocketError) as e:
            raise ProtocolError('Connection broken: %r' % e, e)
        clean_exit = True
    finally:
        if not clean_exit:
            if self._original_response:
                self._original_response.close()
            if self._connection:
                self._connection.close()
        if self._original_response and self._original_response.isclosed():
            self.release_conn()