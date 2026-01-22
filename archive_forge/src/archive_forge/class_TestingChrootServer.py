import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
class TestingChrootServer(chroot.ChrootServer):

    def __init__(self):
        """TestingChrootServer is not usable until start_server is called."""
        super().__init__(None)

    def start_server(self, backing_server=None):
        """Setup the Chroot on backing_server."""
        if backing_server is not None:
            self.backing_transport = transport.get_transport_from_url(backing_server.get_url())
        else:
            self.backing_transport = transport.get_transport_from_path('.')
        super().start_server()

    def get_bogus_url(self):
        raise NotImplementedError