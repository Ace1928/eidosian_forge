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
def test_synchronousBeforeRemovesConspicuouslySimilarDuring(self):
    """
        If a before-phase trigger removes a during-phase trigger which is
        identical to an already-executed before-phase trigger aside from their
        phases, no warning should be emitted and the during-phase trigger
        should not be run.
        """
    events = []

    def trigger():
        events.append('trigger')
    self.event.addTrigger('before', trigger)
    self.event.addTrigger('before', lambda: self.event.removeTrigger(duringTrigger))
    duringTrigger = self.event.addTrigger('during', trigger)
    self.event.fireEvent()
    self.assertEqual(events, ['trigger'])