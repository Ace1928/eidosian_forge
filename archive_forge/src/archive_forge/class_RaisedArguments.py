from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
class RaisedArguments(Exception):
    """
    An exception containing the arguments raised by L{raiser}.
    """

    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs