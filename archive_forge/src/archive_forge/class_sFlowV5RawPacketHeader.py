import struct
import logging
class sFlowV5RawPacketHeader(object):
    _PACK_STR = '!iIII'

    def __init__(self, header_protocol, frame_length, stripped, header_size, header):
        super(sFlowV5RawPacketHeader, self).__init__()
        self.header_protocol = header_protocol
        self.frame_length = frame_length
        self.stripped = stripped
        self.header_size = header_size
        self.header = header

    @classmethod
    def parser(cls, buf, offset):
        header_protocol, frame_length, stripped, header_size = struct.unpack_from(cls._PACK_STR, buf, offset)
        offset += struct.calcsize(cls._PACK_STR)
        header_pack_str = '!%sc' % header_size
        header = struct.unpack_from(header_pack_str, buf, offset)
        msg = cls(header_protocol, frame_length, stripped, header_size, header)
        return msg