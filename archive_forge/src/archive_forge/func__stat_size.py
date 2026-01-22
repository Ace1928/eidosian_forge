import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def _stat_size(self, fp):
    """
        Get the filepath's size as an int

        @param fp: L{twisted.python.filepath.FilePath}
        @return: C{int} representing the size
        """
    return fp.getsize()