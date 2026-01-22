import asyncio
from asyncio import AbstractEventLoop, Queue
from typing import AsyncIterator, Generic, TypeVar
class _MemoryStream(Generic[T]):
    """Stream data from a writer to a reader even if they are in different threads.

    Uses asyncio queues to communicate between two co-routines. This implementation
    should work even if the writer and reader co-routines belong to two different
    event loops (e.g. one running from an event loop in the main thread
    and the other running in an event loop in a background thread).

    This implementation is meant to be used with a single writer and a single reader.

    This is an internal implementation to LangChain please do not use it directly.
    """

    def __init__(self, loop: AbstractEventLoop) -> None:
        """Create a channel for the given loop.

        Args:
            loop: The event loop to use for the channel. The reader is assumed
                  to be running in the same loop as the one passed to this constructor.
                  This will NOT be validated at run time.
        """
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=0)
        self._done = object()

    def get_send_stream(self) -> _SendStream[T]:
        """Get a writer for the channel."""
        return _SendStream[T](reader_loop=self._loop, queue=self._queue, done=self._done)

    def get_receive_stream(self) -> _ReceiveStream[T]:
        """Get a reader for the channel."""
        return _ReceiveStream[T](queue=self._queue, done=self._done)