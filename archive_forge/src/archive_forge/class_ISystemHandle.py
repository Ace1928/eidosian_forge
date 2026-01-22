from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class ISystemHandle(Interface):
    """
    An object that wraps a networking OS-specific handle.
    """

    def getHandle() -> object:
        """
        Return a system- and reactor-specific handle.

        This might be a socket.socket() object, or some other type of
        object, depending on which reactor is being used. Use and
        manipulate at your own risk.

        This might be used in cases where you want to set specific
        options not exposed by the Twisted APIs.
        """