import os
import socket
import traceback
from unittest import skipIf
from zope.interface import implementer
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor
from twisted.internet.tcp import EINPROGRESS, EWOULDBLOCK
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest

                When the problem is detected, the reactor should disconnect this
                file descriptor.  When that happens, stop the reactor so the
                test ends.
                