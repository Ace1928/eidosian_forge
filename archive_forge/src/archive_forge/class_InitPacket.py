import logging
import os
import struct
import time
from pyu2f import errors
from pyu2f import hid
class InitPacket(object):
    """Represent an initial U2FHID packet.

    Represent an initial U2FHID packet.  This packet contains
    metadata necessary to interpret the entire packet stream associated
    with a particular exchange (read or write).

    Attributes:
      packet_size: The size of the hid report (packet) used.  Usually 64.
      cid: The channel id for the connection to the device.
      size: The size of the entire message to be sent (including
          all continuation packets)
      payload: The portion of the message to put into the init packet.
          This must be smaller than packet_size - 7 (the overhead for
          an init packet).
    """

    def __init__(self, packet_size, cid, cmd, size, payload):
        self.packet_size = packet_size
        if len(cid) != 4 or cmd > 255 or size >= 2 ** 16:
            raise errors.InvalidPacketError()
        if len(payload) > self.packet_size - 7:
            raise errors.InvalidPacketError()
        self.cid = cid
        self.cmd = cmd
        self.size = size
        self.payload = payload

    def ToWireFormat(self):
        """Serializes the packet."""
        ret = bytearray(64)
        ret[0:4] = self.cid
        ret[4] = self.cmd
        struct.pack_into('>H', ret, 5, self.size)
        ret[7:7 + len(self.payload)] = self.payload
        return list(map(int, ret))

    @staticmethod
    def FromWireFormat(packet_size, data):
        """Derializes the packet.

      Deserializes the packet from wire format.

      Args:
        packet_size: The size of all packets (usually 64)
        data: List of ints or bytearray containing the data from the wire.

      Returns:
        InitPacket object for specified data

      Raises:
        InvalidPacketError: if the data isn't a valid InitPacket
      """
        ba = bytearray(data)
        if len(ba) != packet_size:
            raise errors.InvalidPacketError()
        cid = ba[0:4]
        cmd = ba[4]
        size = struct.unpack('>H', bytes(ba[5:7]))[0]
        payload = ba[7:7 + size]
        return UsbHidTransport.InitPacket(packet_size, cid, cmd, size, payload)