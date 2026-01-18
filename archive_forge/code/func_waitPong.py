import time
import logging
from threading import Thread, Lock
from yowsup.layers import YowProtocolLayer, YowLayerEvent, EventCallback
from yowsup.common import YowConstants
from yowsup.layers.network import YowNetworkLayer
from yowsup.layers.auth import YowAuthenticationProtocolLayer
from .protocolentities import *
def waitPong(self, id):
    self._pingQueueLock.acquire()
    self._pingQueue[id] = None
    pingQueueSize = len(self._pingQueue)
    self._pingQueueLock.release()
    self.__logger.debug('ping queue size: %d' % pingQueueSize)
    if pingQueueSize >= 2:
        self.getStack().broadcastEvent(YowLayerEvent(YowNetworkLayer.EVENT_STATE_DISCONNECT, reason='Ping Timeout'))