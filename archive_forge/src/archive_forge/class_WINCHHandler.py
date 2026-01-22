import errno
import fcntl
import os
from oslo_log import log as logging
import select
import signal
import socket
import ssl
import struct
import sys
import termios
import time
import tty
from urllib import parse as urlparse
import websocket
from zunclient.common.apiclient import exceptions as acexceptions
from zunclient.common.websocketclient import exceptions
class WINCHHandler(object):
    """WINCH Signal handler

    WINCH Signal handler to keep the PTY correctly sized.
    """

    def __init__(self, client):
        """Initialize a new WINCH handler for the given PTY.

        Initializing a handler has no immediate side-effects. The `start()`
        method must be invoked for the signals to be trapped.
        """
        self.client = client
        self.original_handler = None

    def __enter__(self):
        """Enter

        Invoked on entering a `with` block.
        """
        self.start()
        return self

    def __exit__(self, *_):
        """Exit

        Invoked on exiting a `with` block.
        """
        self.stop()

    def start(self):
        """Start

        Start trapping WINCH signals and resizing the PTY.
        This method saves the previous WINCH handler so it can be restored on
        `stop()`.
        """

        def handle(signum, frame):
            if signum == signal.SIGWINCH:
                LOG.debug('Send command to resize the tty session')
                self.client.handle_resize()
        self.original_handler = signal.signal(signal.SIGWINCH, handle)

    def stop(self):
        """stop

        Stop trapping WINCH signals and restore the previous WINCH handler.
        """
        if self.original_handler is not None:
            signal.signal(signal.SIGWINCH, self.original_handler)