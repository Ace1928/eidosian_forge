from __future__ import absolute_import
import serial
import threading
class FramedPacket(Protocol):
    """
    Read binary packets. Packets are expected to have a start and stop marker.

    The class also keeps track of the transport.
    """
    START = b'('
    STOP = b')'

    def __init__(self):
        self.packet = bytearray()
        self.in_packet = False
        self.transport = None

    def connection_made(self, transport):
        """Store transport"""
        self.transport = transport

    def connection_lost(self, exc):
        """Forget transport"""
        self.transport = None
        self.in_packet = False
        del self.packet[:]
        super(FramedPacket, self).connection_lost(exc)

    def data_received(self, data):
        """Find data enclosed in START/STOP, call handle_packet"""
        for byte in serial.iterbytes(data):
            if byte == self.START:
                self.in_packet = True
            elif byte == self.STOP:
                self.in_packet = False
                self.handle_packet(bytes(self.packet))
                del self.packet[:]
            elif self.in_packet:
                self.packet.extend(byte)
            else:
                self.handle_out_of_packet_data(byte)

    def handle_packet(self, packet):
        """Process packets - to be overridden by subclassing"""
        raise NotImplementedError('please implement functionality in handle_packet')

    def handle_out_of_packet_data(self, data):
        """Process data that is received outside of packets"""
        pass