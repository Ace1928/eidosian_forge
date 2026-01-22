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
def _stat_hardlinks(self, fp):
    """
        Get the number of hardlinks for the filepath - if the number of
        hardlinks is not yet implemented (say in Windows), just return 0 since
        stat-ing a file in Windows seems to return C{st_nlink=0}.

        (Reference:
        U{http://stackoverflow.com/questions/5275731/os-stat-on-windows})

        @param fp: L{twisted.python.filepath.FilePath}
        @return: C{int} representing the number of hardlinks
        """
    try:
        return fp.getNumberOfHardLinks()
    except NotImplementedError:
        return 0