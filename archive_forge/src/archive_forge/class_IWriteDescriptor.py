from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IWriteDescriptor(IFileDescriptor):
    """
    An L{IFileDescriptor} that can write.

    This interface is generally used in conjunction with L{IReactorFDSet}.
    """

    def doWrite() -> Optional[Failure]:
        """
        Some data can be written to your descriptor.

        @return: If an error is encountered which causes the descriptor to
            no longer be valid, a L{Failure} should be returned.  Otherwise,
            L{None}.
        """