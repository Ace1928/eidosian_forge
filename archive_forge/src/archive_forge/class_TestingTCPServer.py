import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
class TestingTCPServer(TestingTCPServerMixin, socketserver.TCPServer):

    def __init__(self, server_address, request_handler_class):
        TestingTCPServerMixin.__init__(self)
        socketserver.TCPServer.__init__(self, server_address, request_handler_class)

    def get_request(self):
        """Get the request and client address from the socket."""
        sock, addr = TestingTCPServerMixin.get_request(self)
        self.clients.append((sock, addr))
        return (sock, addr)

    def shutdown_client(self, client):
        sock, addr = client
        self.shutdown_socket(sock)