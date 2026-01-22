import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
class UDPServer(TCPServer):
    """UDP server class."""
    allow_reuse_address = False
    allow_reuse_port = False
    socket_type = socket.SOCK_DGRAM
    max_packet_size = 8192

    def get_request(self):
        data, client_addr = self.socket.recvfrom(self.max_packet_size)
        return ((data, self.socket), client_addr)

    def server_activate(self):
        pass

    def shutdown_request(self, request):
        self.close_request(request)

    def close_request(self, request):
        pass