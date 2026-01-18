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
def test_tcp_repr(self):
    c = Connector('localhost', 666, object(), 0, object())
    expect = '<twisted.internet.tcp.Connector instance at 0x%x disconnected %s>'
    expect = expect % (id(c), c.getDestination())
    self.assertEqual(repr(c), expect)