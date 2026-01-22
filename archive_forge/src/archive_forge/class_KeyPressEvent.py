import xcffib
import struct
import io
from . import xfixes
from . import xproto
class KeyPressEvent(xcffib.Event):
    xge = True

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Event.__init__(self, unpacker)
        base = unpacker.offset
        self.deviceid, self.time, self.detail, self.root, self.event, self.child, self.root_x, self.root_y, self.event_x, self.event_y, self.buttons_len, self.valuators_len, self.sourceid, self.flags = unpacker.unpack('xx2xHIIIIIiiiiHHH2xI')
        self.mods = ModifierInfo(unpacker)
        unpacker.pad(GroupInfo)
        self.group = GroupInfo(unpacker)
        unpacker.pad('I')
        self.button_mask = xcffib.List(unpacker, 'I', self.buttons_len)
        unpacker.pad('I')
        self.valuator_mask = xcffib.List(unpacker, 'I', self.valuators_len)
        unpacker.pad(FP3232)
        self.axisvalues = xcffib.List(unpacker, FP3232, sum(self.valuator_mask))
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', 2))
        buf.write(struct.pack('=x2xHIIIIIiiiiHHH2xI', self.deviceid, self.time, self.detail, self.root, self.event, self.child, self.root_x, self.root_y, self.event_x, self.event_y, self.buttons_len, self.valuators_len, self.sourceid, self.flags))
        buf.write(self.mods.pack() if hasattr(self.mods, 'pack') else ModifierInfo.synthetic(*self.mods).pack())
        buf.write(self.group.pack() if hasattr(self.group, 'pack') else GroupInfo.synthetic(*self.group).pack())
        buf.write(xcffib.pack_list(self.button_mask, 'I'))
        buf.write(xcffib.pack_list(self.valuator_mask, 'I'))
        buf.write(xcffib.pack_list(self.axisvalues, FP3232))
        buf_len = len(buf.getvalue())
        if buf_len < 32:
            buf.write(struct.pack('x' * (32 - buf_len)))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, deviceid, time, detail, root, event, child, root_x, root_y, event_x, event_y, buttons_len, valuators_len, sourceid, flags, mods, group, button_mask, valuator_mask, axisvalues):
        self = cls.__new__(cls)
        self.deviceid = deviceid
        self.time = time
        self.detail = detail
        self.root = root
        self.event = event
        self.child = child
        self.root_x = root_x
        self.root_y = root_y
        self.event_x = event_x
        self.event_y = event_y
        self.buttons_len = buttons_len
        self.valuators_len = valuators_len
        self.sourceid = sourceid
        self.flags = flags
        self.mods = mods
        self.group = group
        self.button_mask = button_mask
        self.valuator_mask = valuator_mask
        self.axisvalues = axisvalues
        return self