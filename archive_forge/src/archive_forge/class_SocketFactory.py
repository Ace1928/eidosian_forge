from __future__ import annotations
import socket
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar
import trio
class SocketFactory(metaclass=ABCMeta):
    """If you write a custom class implementing the Trio socket interface,
    then you can use a :class:`SocketFactory` to get Trio to use it.

    See :func:`trio.socket.set_custom_socket_factory`.

    """

    @abstractmethod
    def socket(self, family: socket.AddressFamily | int=socket.AF_INET, type: socket.SocketKind | int=socket.SOCK_STREAM, proto: int=0) -> SocketType:
        """Create and return a socket object.

        Your socket object must inherit from :class:`trio.socket.SocketType`,
        which is an empty class whose only purpose is to "mark" which classes
        should be considered valid Trio sockets.

        Called by :func:`trio.socket.socket`.

        Note that unlike :func:`trio.socket.socket`, this does not take a
        ``fileno=`` argument. If a ``fileno=`` is specified, then
        :func:`trio.socket.socket` returns a regular Trio socket object
        instead of calling this method.

        """