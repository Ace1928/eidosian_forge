from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
class SynchronousTestFailureInTearDown(FailureInTearDownMixin, unittest.SynchronousTestCase):
    pass