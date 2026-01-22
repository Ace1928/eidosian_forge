import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
class TestingPathFilteringServer(pathfilter.PathFilteringServer):

    def __init__(self):
        """TestingPathFilteringServer is not usable until start_server
        is called."""

    def start_server(self, backing_server=None):
        """Setup the Chroot on backing_server."""
        if backing_server is not None:
            self.backing_transport = transport.get_transport_from_url(backing_server.get_url())
        else:
            self.backing_transport = transport.get_transport_from_path('.')
        self.backing_transport.clone('added-by-filter').ensure_base()
        self.filter_func = lambda x: 'added-by-filter/' + x
        super().start_server()

    def get_bogus_url(self):
        raise NotImplementedError