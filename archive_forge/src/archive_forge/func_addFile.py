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
def addFile(self, info):
    """
        Append file information dictionary to the list of known files.

        Subclasses can override or extend this method to handle file
        information differently without affecting the parsing of data
        from the server.

        @param info: dictionary containing the parsed representation
                     of the file information
        @type info: dict
        """
    self.files.append(info)