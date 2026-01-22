from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class BoolProto(ProtocolBuffer.ProtocolMessage):
    has_value_ = 0
    value_ = 0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def value(self):
        return self.value_

    def set_value(self, x):
        self.has_value_ = 1
        self.value_ = x

    def clear_value(self):
        if self.has_value_:
            self.has_value_ = 0
            self.value_ = 0

    def has_value(self):
        return self.has_value_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_value():
            self.set_value(x.value())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_value_ != x.has_value_:
            return 0
        if self.has_value_ and self.value_ != x.value_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_value_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: value not set.')
        return initialized

    def ByteSize(self):
        n = 0
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_value_:
            n += 2
        return n

    def Clear(self):
        self.clear_value()

    def OutputUnchecked(self, out):
        out.putVarInt32(8)
        out.putBoolean(self.value_)

    def OutputPartial(self, out):
        if self.has_value_:
            out.putVarInt32(8)
            out.putBoolean(self.value_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_value(d.getBoolean())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_value_:
            res += prefix + 'value: %s\n' % self.DebugFormatBool(self.value_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kvalue = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'value'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.base.BoolProto'