import asyncio
import atexit
import time
import typing as t
from queue import Empty
from threading import Event, Thread
import zmq.asyncio
from jupyter_core.utils import ensure_async
from ._version import protocol_version_info
from .channelsabc import HBChannelABC
from .session import Session
class AsyncZMQSocketChannel(ZMQSocketChannel):
    """A ZMQ socket in an async API"""
    socket: zmq.asyncio.Socket

    def __init__(self, socket: zmq.asyncio.Socket, session: Session, loop: t.Any=None) -> None:
        """Create a channel.

        Parameters
        ----------
        socket : :class:`zmq.asyncio.Socket`
            The ZMQ socket to use.
        session : :class:`session.Session`
            The session to use.
        loop
            Unused here, for other implementations
        """
        if not isinstance(socket, zmq.asyncio.Socket):
            msg = 'Socket must be asyncio'
            raise ValueError(msg)
        super().__init__(socket, session)

    async def _recv(self, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        assert self.socket is not None
        msg = await self.socket.recv_multipart(**kwargs)
        _, smsg = self.session.feed_identities(msg)
        return self.session.deserialize(smsg)

    async def get_msg(self, timeout: t.Optional[float]=None) -> t.Dict[str, t.Any]:
        """Gets a message if there is one that is ready."""
        assert self.socket is not None
        if timeout is not None:
            timeout *= 1000
        ready = await self.socket.poll(timeout)
        if ready:
            res = await self._recv()
            return res
        else:
            raise Empty

    async def get_msgs(self) -> t.List[t.Dict[str, t.Any]]:
        """Get all messages that are currently ready."""
        msgs = []
        while True:
            try:
                msgs.append(await self.get_msg())
            except Empty:
                break
        return msgs

    async def msg_ready(self) -> bool:
        """Is there a message that has been received?"""
        assert self.socket is not None
        return bool(await self.socket.poll(timeout=0))