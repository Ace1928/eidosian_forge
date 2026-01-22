import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
class HTTP1ServerConnection(object):
    """An HTTP/1.x server."""

    def __init__(self, stream: iostream.IOStream, params: Optional[HTTP1ConnectionParameters]=None, context: Optional[object]=None) -> None:
        """
        :arg stream: an `.IOStream`
        :arg params: a `.HTTP1ConnectionParameters` or None
        :arg context: an opaque application-defined object that is accessible
            as ``connection.context``
        """
        self.stream = stream
        if params is None:
            params = HTTP1ConnectionParameters()
        self.params = params
        self.context = context
        self._serving_future = None

    async def close(self) -> None:
        """Closes the connection.

        Returns a `.Future` that resolves after the serving loop has exited.
        """
        self.stream.close()
        assert self._serving_future is not None
        try:
            await self._serving_future
        except Exception:
            pass

    def start_serving(self, delegate: httputil.HTTPServerConnectionDelegate) -> None:
        """Starts serving requests on this connection.

        :arg delegate: a `.HTTPServerConnectionDelegate`
        """
        assert isinstance(delegate, httputil.HTTPServerConnectionDelegate)
        fut = gen.convert_yielded(self._server_request_loop(delegate))
        self._serving_future = fut
        self.stream.io_loop.add_future(fut, lambda f: f.result())

    async def _server_request_loop(self, delegate: httputil.HTTPServerConnectionDelegate) -> None:
        try:
            while True:
                conn = HTTP1Connection(self.stream, False, self.params, self.context)
                request_delegate = delegate.start_request(self, conn)
                try:
                    ret = await conn.read_response(request_delegate)
                except (iostream.StreamClosedError, iostream.UnsatisfiableReadError, asyncio.CancelledError):
                    return
                except _QuietException:
                    conn.close()
                    return
                except Exception:
                    gen_log.error('Uncaught exception', exc_info=True)
                    conn.close()
                    return
                if not ret:
                    return
                await asyncio.sleep(0)
        finally:
            delegate.on_close(self)