import xcffib
import struct
import io
class ModeInfo(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.dotclock, self.hdisplay, self.hsyncstart, self.hsyncend, self.htotal, self.hskew, self.vdisplay, self.vsyncstart, self.vsyncend, self.vtotal, self.flags, self.privsize = unpacker.unpack('IHHHHIHHHH4xI12xI')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IHHHHIHHHH4xI12xI', self.dotclock, self.hdisplay, self.hsyncstart, self.hsyncend, self.htotal, self.hskew, self.vdisplay, self.vsyncstart, self.vsyncend, self.vtotal, self.flags, self.privsize))
        return buf.getvalue()
    fixed_size = 48

    @classmethod
    def synthetic(cls, dotclock, hdisplay, hsyncstart, hsyncend, htotal, hskew, vdisplay, vsyncstart, vsyncend, vtotal, flags, privsize):
        self = cls.__new__(cls)
        self.dotclock = dotclock
        self.hdisplay = hdisplay
        self.hsyncstart = hsyncstart
        self.hsyncend = hsyncend
        self.htotal = htotal
        self.hskew = hskew
        self.vdisplay = vdisplay
        self.vsyncstart = vsyncstart
        self.vsyncend = vsyncend
        self.vtotal = vtotal
        self.flags = flags
        self.privsize = privsize
        return self