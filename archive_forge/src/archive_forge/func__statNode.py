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
def _statNode(self, filePath, keys):
    """
        Shortcut method to get stat info on a node.

        @param filePath: the node to stat.
        @type filePath: C{filepath.FilePath}

        @param keys: the stat keys to get.
        @type keys: C{iterable}
        """
    filePath.restat()
    return [getattr(self, '_stat_' + k)(filePath) for k in keys]