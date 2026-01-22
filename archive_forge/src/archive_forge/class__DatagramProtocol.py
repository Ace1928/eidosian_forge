import asyncio
import socket
import sys
import dns._asyncbackend
import dns._features
import dns.exception
import dns.inet
class _DatagramProtocol:

    def __init__(self):
        self.transport = None
        self.recvfrom = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        if self.recvfrom and (not self.recvfrom.done()):
            self.recvfrom.set_result((data, addr))

    def error_received(self, exc):
        if self.recvfrom and (not self.recvfrom.done()):
            self.recvfrom.set_exception(exc)

    def connection_lost(self, exc):
        if self.recvfrom and (not self.recvfrom.done()):
            if exc is None:
                try:
                    raise EOFError
                except EOFError as e:
                    self.recvfrom.set_exception(e)
            else:
                self.recvfrom.set_exception(exc)

    def close(self):
        self.transport.close()