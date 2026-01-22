import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log
class ProcessAliasTimeout(Exception):
    """
    An error indicating that a timeout occurred while waiting for a process
    to complete.
    """