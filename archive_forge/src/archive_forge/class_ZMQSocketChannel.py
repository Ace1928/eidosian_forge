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
class ZMQSocketChannel:
    """A ZMQ socket wrapper"""

    def __init__(self, socket: zmq.Socket, session: Session, loop: t.Any=None) -> None:
        """Create a channel.

        Parameters
        ----------
        socket : :class:`zmq.Socket`
            The ZMQ socket to use.
        session : :class:`session.Session`
            The session to use.
        loop
            Unused here, for other implementations
        """
        super().__init__()
        self.socket: t.Optional[zmq.Socket] = socket
        self.session = session

    def _recv(self, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        assert self.socket is not None
        msg = self.socket.recv_multipart(**kwargs)
        ident, smsg = self.session.feed_identities(msg)
        return self.session.deserialize(smsg)

    def get_msg(self, timeout: t.Optional[float]=None) -> t.Dict[str, t.Any]:
        """Gets a message if there is one that is ready."""
        assert self.socket is not None
        if timeout is not None:
            timeout *= 1000
        ready = self.socket.poll(timeout)
        if ready:
            res = self._recv()
            return res
        else:
            raise Empty

    def get_msgs(self) -> t.List[t.Dict[str, t.Any]]:
        """Get all messages that are currently ready."""
        msgs = []
        while True:
            try:
                msgs.append(self.get_msg())
            except Empty:
                break
        return msgs

    def msg_ready(self) -> bool:
        """Is there a message that has been received?"""
        assert self.socket is not None
        return bool(self.socket.poll(timeout=0))

    def close(self) -> None:
        """Close the socket channel."""
        if self.socket is not None:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
            self.socket = None
    stop = close

    def is_alive(self) -> bool:
        """Test whether the channel is alive."""
        return self.socket is not None

    def send(self, msg: t.Dict[str, t.Any]) -> None:
        """Pass a message to the ZMQ socket to send"""
        assert self.socket is not None
        self.session.send(self.socket, msg)

    def start(self) -> None:
        """Start the socket channel."""
        pass