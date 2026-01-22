from __future__ import annotations
import sys
from zope.interface import implementer
from CFNetwork import (
from CoreFoundation import (
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker
def _flag2idx(self, flag):
    """
        Convert a C{kCFSocket...} constant to an index into the read/write
        state list (C{_READ} or C{_WRITE}) (the 4th element of the value of
        C{self._fdmap}).

        @param flag: C{kCFSocketReadCallBack} or C{kCFSocketWriteCallBack}

        @return: C{_READ} or C{_WRITE}
        """
    return {kCFSocketReadCallBack: _READ, kCFSocketWriteCallBack: _WRITE}[flag]