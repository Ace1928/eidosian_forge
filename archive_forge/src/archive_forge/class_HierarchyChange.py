import xcffib
import struct
import io
from . import xfixes
from . import xproto
class HierarchyChange(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.type, self.len = unpacker.unpack('HH')
        if self.type == HierarchyChangeType.AddMaster:
            self.name_len, self.send_core, self.enable = unpacker.unpack('HBB')
            self.name = xcffib.List(unpacker, 'c', self.name_len)
        if self.type == HierarchyChangeType.RemoveMaster:
            self.deviceid, self.return_mode, self.return_pointer, self.return_keyboard = unpacker.unpack('HBxHH')
        if self.type == HierarchyChangeType.AttachSlave:
            self.deviceid, self.master = unpacker.unpack('HH')
        if self.type == HierarchyChangeType.DetachSlave:
            self.deviceid, = unpacker.unpack('H2x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HH', self.type, self.len))
        if self.type & HierarchyChangeType.AddMaster:
            self.name_len = self.data.pop(0)
            self.send_core = self.data.pop(0)
            self.enable = self.data.pop(0)
            self.name = self.data.pop(0)
            self.data.pop(0)
            buf.write(struct.pack('=HBB', self.name_len, self.send_core, self.enable))
            buf.write(xcffib.pack_list(self.name, 'c'))
            buf.write(struct.pack('=4x'))
        if self.type & HierarchyChangeType.RemoveMaster:
            self.deviceid = self.data.pop(0)
            self.return_mode = self.data.pop(0)
            self.return_pointer = self.data.pop(0)
            self.return_keyboard = self.data.pop(0)
            buf.write(struct.pack('=HBxHH', self.deviceid, self.return_mode, self.return_pointer, self.return_keyboard))
        if self.type & HierarchyChangeType.AttachSlave:
            self.deviceid = self.data.pop(0)
            self.master = self.data.pop(0)
            buf.write(struct.pack('=HH', self.deviceid, self.master))
        if self.type & HierarchyChangeType.DetachSlave:
            self.deviceid = self.data.pop(0)
            buf.write(struct.pack('=H2x', self.deviceid))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, type, len, data):
        self = cls.__new__(cls)
        self.type = type
        self.len = len
        self.data = data
        return self