import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
def InternalRecv(self):
    """Receives a message from the device, including defragmenting it."""
    first_read = self.InternalReadFrame()
    first_packet = UsbHidTransport.InitPacket.FromWireFormat(self.packet_size, first_read)
    data = first_packet.payload
    to_read = first_packet.size - len(first_packet.payload)
    seq = 0
    while to_read > 0:
        next_read = self.InternalReadFrame()
        next_packet = UsbHidTransport.ContPacket.FromWireFormat(self.packet_size, next_read)
        if self.cid != next_packet.cid:
            continue
        if seq != next_packet.seq:
            raise errors.HardwareError('Packets received out of order')
        to_read -= len(next_packet.payload)
        data.extend(next_packet.payload)
        seq += 1
    data = data[0:first_packet.size]
    return (first_packet.cmd, data)