import xcffib
import struct
import io
from . import xproto
from . import shm
class ImageFormatInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.id, self.type, self.byte_order = unpacker.unpack('IBB2x')
        self.guid = xcffib.List(unpacker, 'B', 16)
        self.bpp, self.num_planes, self.depth, self.red_mask, self.green_mask, self.blue_mask, self.format, self.y_sample_bits, self.u_sample_bits, self.v_sample_bits, self.vhorz_y_period, self.vhorz_u_period, self.vhorz_v_period, self.vvert_y_period, self.vvert_u_period, self.vvert_v_period = unpacker.unpack('BB2xB3xIIIB3xIIIIIIIII')
        unpacker.pad('B')
        self.vcomp_order = xcffib.List(unpacker, 'B', 32)
        self.vscanline_order, = unpacker.unpack('B11x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IBB2x', self.id, self.type, self.byte_order))
        buf.write(xcffib.pack_list(self.guid, 'B'))
        buf.write(struct.pack('=B', self.bpp))
        buf.write(struct.pack('=B', self.num_planes))
        buf.write(struct.pack('=2x'))
        buf.write(struct.pack('=B', self.depth))
        buf.write(struct.pack('=3x'))
        buf.write(struct.pack('=I', self.red_mask))
        buf.write(struct.pack('=I', self.green_mask))
        buf.write(struct.pack('=I', self.blue_mask))
        buf.write(struct.pack('=B', self.format))
        buf.write(struct.pack('=3x'))
        buf.write(struct.pack('=I', self.y_sample_bits))
        buf.write(struct.pack('=I', self.u_sample_bits))
        buf.write(struct.pack('=I', self.v_sample_bits))
        buf.write(struct.pack('=I', self.vhorz_y_period))
        buf.write(struct.pack('=I', self.vhorz_u_period))
        buf.write(struct.pack('=I', self.vhorz_v_period))
        buf.write(struct.pack('=I', self.vvert_y_period))
        buf.write(struct.pack('=I', self.vvert_u_period))
        buf.write(struct.pack('=I', self.vvert_v_period))
        buf.write(xcffib.pack_list(self.vcomp_order, 'B'))
        buf.write(struct.pack('=B', self.vscanline_order))
        buf.write(struct.pack('=11x'))
        return buf.getvalue()
    fixed_size = 128

    @classmethod
    def synthetic(cls, id, type, byte_order, guid, bpp, num_planes, depth, red_mask, green_mask, blue_mask, format, y_sample_bits, u_sample_bits, v_sample_bits, vhorz_y_period, vhorz_u_period, vhorz_v_period, vvert_y_period, vvert_u_period, vvert_v_period, vcomp_order, vscanline_order):
        self = cls.__new__(cls)
        self.id = id
        self.type = type
        self.byte_order = byte_order
        self.guid = guid
        self.bpp = bpp
        self.num_planes = num_planes
        self.depth = depth
        self.red_mask = red_mask
        self.green_mask = green_mask
        self.blue_mask = blue_mask
        self.format = format
        self.y_sample_bits = y_sample_bits
        self.u_sample_bits = u_sample_bits
        self.v_sample_bits = v_sample_bits
        self.vhorz_y_period = vhorz_y_period
        self.vhorz_u_period = vhorz_u_period
        self.vhorz_v_period = vhorz_v_period
        self.vvert_y_period = vvert_y_period
        self.vvert_u_period = vvert_u_period
        self.vvert_v_period = vvert_v_period
        self.vcomp_order = vcomp_order
        self.vscanline_order = vscanline_order
        return self