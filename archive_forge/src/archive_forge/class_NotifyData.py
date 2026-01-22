import xcffib
import struct
import io
from . import xproto
from . import render
class NotifyData(xcffib.Union):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Union.__init__(self, unpacker)
        self.cc = CrtcChange(unpacker.copy())
        self.oc = OutputChange(unpacker.copy())
        self.op = OutputProperty(unpacker.copy())
        self.pc = ProviderChange(unpacker.copy())
        self.pp = ProviderProperty(unpacker.copy())
        self.rc = ResourceChange(unpacker.copy())
        self.lc = LeaseNotify(unpacker.copy())

    def pack(self):
        buf = io.BytesIO()
        buf.write(self.cc.pack() if hasattr(self.cc, 'pack') else CrtcChange.synthetic(*self.cc).pack())
        return buf.getvalue()