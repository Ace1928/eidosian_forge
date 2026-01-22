import xcffib
import struct
import io
class GetModeLineReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        self.dotclock, self.hdisplay, self.hsyncstart, self.hsyncend, self.htotal, self.hskew, self.vdisplay, self.vsyncstart, self.vsyncend, self.vtotal, self.flags, self.privsize = unpacker.unpack('xx2x4xIHHHHHHHHH2xI12xI')
        self.private = xcffib.List(unpacker, 'B', self.privsize)
        self.bufsize = unpacker.offset - base