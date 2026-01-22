import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class EventReactor:
    """
    Partial L{IReactorCore} implementation with simple event-related
    methods.

    @ivar _running: A C{bool} indicating whether the reactor is pretending
        to have been started already or not.

    @ivar triggers: A C{list} of pending system event triggers.
    """

    def __init__(self, running):
        self._running = running
        self.triggers = []

    def callWhenRunning(self, function):
        if self._running:
            function()
        else:
            return self.addSystemEventTrigger('after', 'startup', function)

    def addSystemEventTrigger(self, phase, event, trigger):
        handle = (phase, event, trigger)
        self.triggers.append(handle)
        return handle

    def removeSystemEventTrigger(self, handle):
        self.triggers.remove(handle)