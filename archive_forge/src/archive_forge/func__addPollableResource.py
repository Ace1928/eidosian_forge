from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
def _addPollableResource(self, res):
    self._resources.append(res)
    self._checkPollingState()