import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def addSystemEventTrigger(self, phase, event, trigger):
    handle = (phase, event, trigger)
    self.triggers.append(handle)
    return handle