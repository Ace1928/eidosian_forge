from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
def tryRegister() -> None:
    exc = self.assertRaises(ReactorAlreadyRunning, reactor.registerGApplication, app)
    self.assertEqual(exc.args[0], "Can't register application after reactor was started.")
    reactor.stop()