from __future__ import annotations
import sys
from zope.interface import implementer
from CFNetwork import (
from CoreFoundation import (
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker

        Emulate the behavior of C{iterate()} for things that want to call it,
        by letting the loop run for a little while and then scheduling a timed
        call to exit it.
        