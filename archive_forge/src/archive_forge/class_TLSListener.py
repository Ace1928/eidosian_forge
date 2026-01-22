from __future__ import annotations
import logging
import re
import ssl
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, Tuple, TypeVar
from .. import (
from .._core._typedattr import TypedAttributeSet, typed_attribute
from ..abc import AnyByteStream, ByteStream, Listener, TaskGroup
@dataclass(eq=False)
class TLSListener(Listener[TLSStream]):
    """
    A convenience listener that wraps another listener and auto-negotiates a TLS session
    on every accepted connection.

    If the TLS handshake times out or raises an exception,
    :meth:`handle_handshake_error` is called to do whatever post-mortem processing is
    deemed necessary.

    Supports only the :attr:`~TLSAttribute.standard_compatible` extra attribute.

    :param Listener listener: the listener to wrap
    :param ssl_context: the SSL context object
    :param standard_compatible: a flag passed through to :meth:`TLSStream.wrap`
    :param handshake_timeout: time limit for the TLS handshake
        (passed to :func:`~anyio.fail_after`)
    """
    listener: Listener[Any]
    ssl_context: ssl.SSLContext
    standard_compatible: bool = True
    handshake_timeout: float = 30

    @staticmethod
    async def handle_handshake_error(exc: BaseException, stream: AnyByteStream) -> None:
        """
        Handle an exception raised during the TLS handshake.

        This method does 3 things:

        #. Forcefully closes the original stream
        #. Logs the exception (unless it was a cancellation exception) using the
           ``anyio.streams.tls`` logger
        #. Reraises the exception if it was a base exception or a cancellation exception

        :param exc: the exception
        :param stream: the original stream

        """
        await aclose_forcefully(stream)
        if not isinstance(exc, get_cancelled_exc_class()):
            logging.getLogger(__name__).exception('Error during TLS handshake', exc_info=exc)
        if not isinstance(exc, Exception) or isinstance(exc, get_cancelled_exc_class()):
            raise

    async def serve(self, handler: Callable[[TLSStream], Any], task_group: TaskGroup | None=None) -> None:

        @wraps(handler)
        async def handler_wrapper(stream: AnyByteStream) -> None:
            from .. import fail_after
            try:
                with fail_after(self.handshake_timeout):
                    wrapped_stream = await TLSStream.wrap(stream, ssl_context=self.ssl_context, standard_compatible=self.standard_compatible)
            except BaseException as exc:
                await self.handle_handshake_error(exc, stream)
            else:
                await handler(wrapped_stream)
        await self.listener.serve(handler_wrapper, task_group)

    async def aclose(self) -> None:
        await self.listener.aclose()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return {TLSAttribute.standard_compatible: lambda: self.standard_compatible}