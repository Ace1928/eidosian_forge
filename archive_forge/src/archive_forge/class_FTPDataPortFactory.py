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
class FTPDataPortFactory(protocol.ServerFactory):
    """
    Factory for data connections that use the PORT command

    (i.e. "active" transfers)
    """
    noisy = False

    def buildProtocol(self, addr):
        self.protocol.factory = self
        self.port.loseConnection()
        return self.protocol