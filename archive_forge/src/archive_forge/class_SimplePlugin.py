import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
class SimplePlugin(object):
    """Plugin base class which auto-subscribes methods for known channels."""
    bus = None
    'A :class:`Bus <cherrypy.process.wspbus.Bus>`, usually cherrypy.engine.\n    '

    def __init__(self, bus):
        self.bus = bus

    def subscribe(self):
        """Register this object as a (multi-channel) listener on the bus."""
        for channel in self.bus.listeners:
            method = getattr(self, channel, None)
            if method is not None:
                self.bus.subscribe(channel, method)

    def unsubscribe(self):
        """Unregister this object as a listener on the bus."""
        for channel in self.bus.listeners:
            method = getattr(self, channel, None)
            if method is not None:
                self.bus.unsubscribe(channel, method)