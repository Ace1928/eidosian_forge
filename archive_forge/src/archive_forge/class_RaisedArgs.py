import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class RaisedArgs(Exception):
    """
    An exception which can be raised by fakes to test that the fake is called
    with expected arguments.
    """

    def __init__(self, args, kwargs):
        """
        Store the positional and keyword arguments as attributes.

        @param args: The positional args.
        @param kwargs: The keyword args.
        """
        self.args = args
        self.kwargs = kwargs