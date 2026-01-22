import xcffib
import struct
import io
class GetDeviceInfoReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.framebuffer_handle_low, self.framebuffer_handle_high, self.framebuffer_origin_offset, self.framebuffer_size, self.framebuffer_stride, self.device_private_size = unpacker.unpack('xx2x4xIIIIII')
        self.device_private = xcffib.List(unpacker, 'I', self.device_private_size)
        self.bufsize = unpacker.offset - base