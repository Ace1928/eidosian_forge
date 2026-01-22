from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
def _pollEvent(self):
    workUnits = 0.0
    anyActive = []
    for resource in self._resources:
        if resource.active:
            workUnits += resource.checkWork()
            if resource.active:
                anyActive.append(resource)
    newTimeout = self._currentTimeout
    if workUnits:
        newTimeout = self._currentTimeout / (workUnits + 1.0)
        if newTimeout < MIN_TIMEOUT:
            newTimeout = MIN_TIMEOUT
    else:
        newTimeout = self._currentTimeout * 2.0
        if newTimeout > MAX_TIMEOUT:
            newTimeout = MAX_TIMEOUT
    self._currentTimeout = newTimeout
    if anyActive:
        self._pollTimer = self._reschedule()