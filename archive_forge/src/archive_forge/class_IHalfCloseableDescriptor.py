from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IHalfCloseableDescriptor(Interface):
    """
    A descriptor that can be half-closed.
    """

    def writeConnectionLost(reason: Failure) -> None:
        """
        Indicates write connection was lost.
        """

    def readConnectionLost(reason: Failure) -> None:
        """
        Indicates read connection was lost.
        """