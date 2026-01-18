import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
def test_reentrantStreamingProducerUsingWriteSequence(self):
    """
        Like L{test_reentrantStreamingProducerUsingWrite}, but for calls to
        C{writeSequence}.

        C{writeSequence} is B{not} part of L{IConsumer}, however
        C{abstract.FileDescriptor} has supported consumery behavior in response
        to calls to C{writeSequence} forever.
        """
    return self._reentrantStreamingProducerTest('writeSequence')