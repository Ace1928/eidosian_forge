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
class MessageStandardEncodingTests(StandardEncodingTestsMixin, unittest.SynchronousTestCase):
    """
    Tests for the encoding and decoding of various standard (non-EDNS) messages
    by L{dns.Message}.
    """

    @staticmethod
    def messageFactory(**kwargs):
        """
        This function adapts constructor arguments expected by
        _EDNSMessage.__init__ to arguments suitable for use with the
        Message.__init__.

        Also handles the fact that unlike L{dns._EDNSMessage},
        L{dns.Message.__init__} does not accept queries, answers etc as
        arguments.

        Also removes any L{dns._EDNSMessage} specific arguments.

        @param args: The positional arguments which will be passed to
            L{dns.Message.__init__}.

        @param kwargs: The keyword arguments which will be stripped of EDNS
            specific arguments before being passed to L{dns.Message.__init__}.

        @return: An L{dns.Message} instance.
        """
        queries = kwargs.pop('queries', [])
        answers = kwargs.pop('answers', [])
        authority = kwargs.pop('authority', [])
        additional = kwargs.pop('additional', [])
        kwargs.pop('ednsVersion', None)
        m = dns.Message(**kwargs)
        m.queries = queries
        m.answers = answers
        m.authority = authority
        m.additional = additional
        return MessageComparable(m)