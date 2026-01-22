from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.defer import Deferred, TimeoutError, gatherResults, succeed
from twisted.internet.interfaces import IResolverSimple
from twisted.names import client, root
from twisted.names.dns import (
from twisted.names.error import DNSNameError, ResolverError
from twisted.names.root import Resolver
from twisted.names.test.test_util import MemoryReactor
from twisted.python.log import msg
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase, TestCase
def _respond(self, answers=[], authority=[], additional=[], rCode=OK):
    """
        Create a L{Message} suitable for use as a response to a query.

        @param answers: A C{list} of two-tuples giving data for the answers
            section of the message.  The first element of each tuple is a name
            for the L{RRHeader}.  The second element is the payload.
        @param authority: A C{list} like C{answers}, but for the authority
            section of the response.
        @param additional: A C{list} like C{answers}, but for the
            additional section of the response.
        @param rCode: The response code the message will be created with.

        @return: A new L{Message} initialized with the given values.
        """
    response = Message(rCode=rCode)
    for section, data in [(response.answers, answers), (response.authority, authority), (response.additional, additional)]:
        section.extend([RRHeader(name, record.TYPE, getattr(record, 'CLASS', IN), payload=record) for name, record in data])
    return response