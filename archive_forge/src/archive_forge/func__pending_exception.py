import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def _pending_exception(self, thread):
    for sock, addr, connection_thread in self.clients:
        if connection_thread is not None:
            connection_thread.pending_exception()
    TestingTCPServerMixin._pending_exception(self, thread)