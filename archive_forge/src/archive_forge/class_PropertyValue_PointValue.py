from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class PropertyValue_PointValue(ProtocolBuffer.ProtocolMessage):
    has_x_ = 0
    x_ = 0.0
    has_y_ = 0
    y_ = 0.0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def x(self):
        return self.x_

    def set_x(self, x):
        self.has_x_ = 1
        self.x_ = x

    def clear_x(self):
        if self.has_x_:
            self.has_x_ = 0
            self.x_ = 0.0

    def has_x(self):
        return self.has_x_

    def y(self):
        return self.y_

    def set_y(self, x):
        self.has_y_ = 1
        self.y_ = x

    def clear_y(self):
        if self.has_y_:
            self.has_y_ = 0
            self.y_ = 0.0

    def has_y(self):
        return self.has_y_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_x():
            self.set_x(x.x())
        if x.has_y():
            self.set_y(x.y())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_x_ != x.has_x_:
            return 0
        if self.has_x_ and self.x_ != x.x_:
            return 0
        if self.has_y_ != x.has_y_:
            return 0
        if self.has_y_ and self.y_ != x.y_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_x_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: x not set.')
        if not self.has_y_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: y not set.')
        return initialized

    def ByteSize(self):
        n = 0
        return n + 18

    def ByteSizePartial(self):
        n = 0
        if self.has_x_:
            n += 9
        if self.has_y_:
            n += 9
        return n

    def Clear(self):
        self.clear_x()
        self.clear_y()

    def OutputUnchecked(self, out):
        out.putVarInt32(49)
        out.putDouble(self.x_)
        out.putVarInt32(57)
        out.putDouble(self.y_)

    def OutputPartial(self, out):
        if self.has_x_:
            out.putVarInt32(49)
            out.putDouble(self.x_)
        if self.has_y_:
            out.putVarInt32(57)
            out.putDouble(self.y_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 44:
                break
            if tt == 49:
                self.set_x(d.getDouble())
                continue
            if tt == 57:
                self.set_y(d.getDouble())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_x_:
            res += prefix + 'x: %s\n' % self.DebugFormat(self.x_)
        if self.has_y_:
            res += prefix + 'y: %s\n' % self.DebugFormat(self.y_)
        return res