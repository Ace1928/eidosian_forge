from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReadDescriptor(IFileDescriptor):
    """
    An L{IFileDescriptor} that can read.

    This interface is generally used in conjunction with L{IReactorFDSet}.
    """

    def doRead() -> Optional[Failure]:
        """
        Some data is available for reading on your descriptor.

        @return: If an error is encountered which causes the descriptor to
            no longer be valid, a L{Failure} should be returned.  Otherwise,
            L{None}.
        """