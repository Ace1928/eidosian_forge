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
def test_synchronousRemovePendingDuring(self):
    """
        If a during-phase trigger removes another during-phase trigger which
        has not yet run, the removed trigger should not be run.
        """
    events = []
    self.event.addTrigger('during', lambda: self.event.removeTrigger(duringHandle))
    duringHandle = self.event.addTrigger('during', events.append, ('first', 'during'))
    self.event.addTrigger('during', events.append, ('second', 'during'))
    self.event.fireEvent()
    self.assertEqual(events, [('second', 'during')])