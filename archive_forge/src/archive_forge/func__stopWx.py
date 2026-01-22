from queue import Empty, Queue
from twisted.internet import _threadedselect
from twisted.python import log, runtime
def _stopWx(self):
    """
        Stop the wx event loop if it hasn't already been stopped.

        Called during Twisted event loop shutdown.
        """
    if hasattr(self, 'wxapp'):
        self.wxapp.ExitMainLoop()