from __future__ import annotations
import math
from typing import TYPE_CHECKING, Protocol
import attrs
import trio
from . import _core
from ._core import Abort, ParkingLot, RaiseCancelT, enable_ki_protection
from ._util import final
@final
class StrictFIFOLock(_LockImpl):
    """A variant of :class:`Lock` where tasks are guaranteed to acquire the
    lock in strict first-come-first-served order.

    An example of when this is useful is if you're implementing something like
    :class:`trio.SSLStream` or an HTTP/2 server using `h2
    <https://hyper-h2.readthedocs.io/>`__, where you have multiple concurrent
    tasks that are interacting with a shared state machine, and at
    unpredictable moments the state machine requests that a chunk of data be
    sent over the network. (For example, when using h2 simply reading incoming
    data can occasionally `create outgoing data to send
    <https://http2.github.io/http2-spec/#PING>`__.) The challenge is to make
    sure that these chunks are sent in the correct order, without being
    garbled.

    One option would be to use a regular :class:`Lock`, and wrap it around
    every interaction with the state machine::

        # This approach is sometimes workable but often sub-optimal; see below
        async with lock:
            state_machine.do_something()
            if state_machine.has_data_to_send():
                await conn.sendall(state_machine.get_data_to_send())

    But this can be problematic. If you're using h2 then *usually* reading
    incoming data doesn't create the need to send any data, so we don't want
    to force every task that tries to read from the network to sit and wait
    a potentially long time for ``sendall`` to finish. And in some situations
    this could even potentially cause a deadlock, if the remote peer is
    waiting for you to read some data before it accepts the data you're
    sending.

    :class:`StrictFIFOLock` provides an alternative. We can rewrite our
    example like::

        # Note: no awaits between when we start using the state machine and
        # when we block to take the lock!
        state_machine.do_something()
        if state_machine.has_data_to_send():
            # Notice that we fetch the data to send out of the state machine
            # *before* sleeping, so that other tasks won't see it.
            chunk = state_machine.get_data_to_send()
            async with strict_fifo_lock:
                await conn.sendall(chunk)

    First we do all our interaction with the state machine in a single
    scheduling quantum (notice there are no ``await``\\s in there), so it's
    automatically atomic with respect to other tasks. And then if and only if
    we have data to send, we get in line to send it – and
    :class:`StrictFIFOLock` guarantees that each task will send its data in
    the same order that the state machine generated it.

    Currently, :class:`StrictFIFOLock` is identical to :class:`Lock`,
    but (a) this may not always be true in the future, especially if Trio ever
    implements `more sophisticated scheduling policies
    <https://github.com/python-trio/trio/issues/32>`__, and (b) the above code
    is relying on a pretty subtle property of its lock. Using a
    :class:`StrictFIFOLock` acts as an executable reminder that you're relying
    on this property.

    """