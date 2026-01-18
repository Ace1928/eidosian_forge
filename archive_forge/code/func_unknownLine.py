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
def unknownLine(self, line):
    """
        Deal with received lines which could not be parsed as file
        information.

        Subclasses can override this to perform any special processing
        needed.

        @param line: unparsable line as received
        @type line: str
        """
    pass