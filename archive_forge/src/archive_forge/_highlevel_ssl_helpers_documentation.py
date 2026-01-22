from __future__ import annotations
import ssl
from typing import TYPE_CHECKING, NoReturn, TypeVar
import trio
from ._highlevel_open_tcp_stream import DEFAULT_DELAY
Listen for incoming TCP connections, and for each one start a task
    running ``handler(stream)``.

    This is a thin convenience wrapper around
    :func:`open_ssl_over_tcp_listeners` and :func:`serve_listeners` – see them
    for full details.

    .. warning::

       If ``handler`` raises an exception, then this function doesn't do
       anything special to catch it – so by default the exception will
       propagate out and crash your server. If you don't want this, then catch
       exceptions inside your ``handler``, or use a ``handler_nursery`` object
       that responds to exceptions in some other way.

    When used with ``nursery.start`` you get back the newly opened listeners.
    See the documentation for :func:`serve_tcp` for an example where this is
    useful.

    Args:
      handler: The handler to start for each incoming connection. Passed to
          :func:`serve_listeners`.

      port (int): The port to listen on. Use 0 to let the kernel pick
          an open port. Ultimately passed to :func:`open_tcp_listeners`.

      ssl_context (~ssl.SSLContext): The SSL context to use for all incoming
          connections. Passed to :func:`open_ssl_over_tcp_listeners`.

      host (str, bytes, or None): The address to bind to; use ``None`` to bind
          to the wildcard address. Ultimately passed to
          :func:`open_tcp_listeners`.

      https_compatible (bool): Set this to True if you want to use
          "HTTPS-style" TLS. See :class:`~trio.SSLStream` for details.

      backlog (int or None): See :class:`~trio.SSLStream` for details.

      handler_nursery: The nursery to start handlers in, or None to use an
          internal nursery. Passed to :func:`serve_listeners`.

      task_status: This function can be used with ``nursery.start``.

    Returns:
      This function only returns when cancelled.

    