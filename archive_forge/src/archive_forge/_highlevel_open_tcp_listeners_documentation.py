from __future__ import annotations
import errno
import math
import sys
from typing import TYPE_CHECKING
import trio
from trio import TaskStatus
from . import socket as tsocket
from ._deprecate import warn_deprecated
Listen for incoming TCP connections, and for each one start a task
    running ``handler(stream)``.

    This is a thin convenience wrapper around :func:`open_tcp_listeners` and
    :func:`serve_listeners` – see them for full details.

    .. warning::

       If ``handler`` raises an exception, then this function doesn't do
       anything special to catch it – so by default the exception will
       propagate out and crash your server. If you don't want this, then catch
       exceptions inside your ``handler``, or use a ``handler_nursery`` object
       that responds to exceptions in some other way.

    When used with ``nursery.start`` you get back the newly opened listeners.
    So, for example, if you want to start a server in your test suite and then
    connect to it to check that it's working properly, you can use something
    like::

        from trio import SocketListener, SocketStream
        from trio.testing import open_stream_to_socket_listener

        async with trio.open_nursery() as nursery:
            listeners: list[SocketListener] = await nursery.start(serve_tcp, handler, 0)
            client_stream: SocketStream = await open_stream_to_socket_listener(listeners[0])

            # Then send and receive data on 'client_stream', for example:
            await client_stream.send_all(b"GET / HTTP/1.0\r\n\r\n")

    This avoids several common pitfalls:

    1. It lets the kernel pick a random open port, so your test suite doesn't
       depend on any particular port being open.

    2. It waits for the server to be accepting connections on that port before
       ``start`` returns, so there's no race condition where the incoming
       connection arrives before the server is ready.

    3. It uses the Listener object to find out which port was picked, so it
       can connect to the right place.

    Args:
      handler: The handler to start for each incoming connection. Passed to
          :func:`serve_listeners`.

      port: The port to listen on. Use 0 to let the kernel pick an open port.
          Passed to :func:`open_tcp_listeners`.

      host (str, bytes, or None): The host interface to listen on; use
          ``None`` to bind to the wildcard address. Passed to
          :func:`open_tcp_listeners`.

      backlog: The listen backlog, or None to have a good default picked.
          Passed to :func:`open_tcp_listeners`.

      handler_nursery: The nursery to start handlers in, or None to use an
          internal nursery. Passed to :func:`serve_listeners`.

      task_status: This function can be used with ``nursery.start``.

    Returns:
      This function only returns when cancelled.

    