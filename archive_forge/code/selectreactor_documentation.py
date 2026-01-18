import select
import sys
from errno import EBADF, EINTR
from time import sleep
from typing import Type
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
from twisted.python.runtime import platformType

        Remove a Selectable for notification of data available to write.
        