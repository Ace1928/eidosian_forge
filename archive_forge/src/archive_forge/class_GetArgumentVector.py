import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
class GetArgumentVector(UtilityProcessProtocol):
    """
    Protocol which will read a serialized argv from a process and
    expose it to interested parties.
    """
    programName = b'twisted.test.process_getargv'

    def parseChunks(self, chunks):
        """
        Parse the output from the process to which this protocol was
        connected, which is a single unterminated line of \\0-separated
        strings giving the argv of that process.  Return this as a list of
        str objects.
        """
        return b''.join(chunks).split(b'\x00')