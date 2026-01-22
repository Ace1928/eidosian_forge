from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IReadWriteDescriptor(IReadDescriptor, IWriteDescriptor):
    """
    An L{IFileDescriptor} that can both read and write.
    """