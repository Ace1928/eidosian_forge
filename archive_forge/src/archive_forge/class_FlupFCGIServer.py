import os
import sys
import time
import warnings
import contextlib
import portend
class FlupFCGIServer(object):
    """Adapter for a flup.server.fcgi.WSGIServer."""

    def __init__(self, *args, **kwargs):
        if kwargs.get('bindAddress', None) is None:
            import socket
            if not hasattr(socket, 'fromfd'):
                raise ValueError('Dynamic FCGI server not available on this platform. You must use a static or external one by providing a legal bindAddress.')
        self.args = args
        self.kwargs = kwargs
        self.ready = False

    def start(self):
        """Start the FCGI server."""
        from flup.server.fcgi import WSGIServer
        self.fcgiserver = WSGIServer(*self.args, **self.kwargs)
        self.fcgiserver._installSignalHandlers = lambda: None
        self.fcgiserver._oldSIGs = []
        self.ready = True
        self.fcgiserver.run()

    def stop(self):
        """Stop the HTTP server."""
        self.fcgiserver._keepGoing = False
        self.fcgiserver._threadPool.maxSpare = self.fcgiserver._threadPool._idleCount
        self.ready = False