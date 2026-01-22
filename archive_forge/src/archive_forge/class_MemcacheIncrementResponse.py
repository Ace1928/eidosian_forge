from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheIncrementResponse(ProtocolBuffer.ProtocolMessage):
    OK = 1
    NOT_CHANGED = 2
    ERROR = 3
    DEADLINE_EXCEEDED = 4
    UNREACHABLE = 5
    OTHER_ERROR = 6
    _IncrementStatusCode_NAMES = {1: 'OK', 2: 'NOT_CHANGED', 3: 'ERROR', 4: 'DEADLINE_EXCEEDED', 5: 'UNREACHABLE', 6: 'OTHER_ERROR'}

    def IncrementStatusCode_Name(cls, x):
        return cls._IncrementStatusCode_NAMES.get(x, '')
    IncrementStatusCode_Name = classmethod(IncrementStatusCode_Name)
    has_new_value_ = 0
    new_value_ = 0
    has_increment_status_ = 0
    increment_status_ = 0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def new_value(self):
        return self.new_value_

    def set_new_value(self, x):
        self.has_new_value_ = 1
        self.new_value_ = x

    def clear_new_value(self):
        if self.has_new_value_:
            self.has_new_value_ = 0
            self.new_value_ = 0

    def has_new_value(self):
        return self.has_new_value_

    def increment_status(self):
        return self.increment_status_

    def set_increment_status(self, x):
        self.has_increment_status_ = 1
        self.increment_status_ = x

    def clear_increment_status(self):
        if self.has_increment_status_:
            self.has_increment_status_ = 0
            self.increment_status_ = 0

    def has_increment_status(self):
        return self.has_increment_status_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_new_value():
            self.set_new_value(x.new_value())
        if x.has_increment_status():
            self.set_increment_status(x.increment_status())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_new_value_ != x.has_new_value_:
            return 0
        if self.has_new_value_ and self.new_value_ != x.new_value_:
            return 0
        if self.has_increment_status_ != x.has_increment_status_:
            return 0
        if self.has_increment_status_ and self.increment_status_ != x.increment_status_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_new_value_:
            n += 1 + self.lengthVarInt64(self.new_value_)
        if self.has_increment_status_:
            n += 1 + self.lengthVarInt64(self.increment_status_)
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_new_value_:
            n += 1 + self.lengthVarInt64(self.new_value_)
        if self.has_increment_status_:
            n += 1 + self.lengthVarInt64(self.increment_status_)
        return n

    def Clear(self):
        self.clear_new_value()
        self.clear_increment_status()

    def OutputUnchecked(self, out):
        if self.has_new_value_:
            out.putVarInt32(8)
            out.putVarUint64(self.new_value_)
        if self.has_increment_status_:
            out.putVarInt32(16)
            out.putVarInt32(self.increment_status_)

    def OutputPartial(self, out):
        if self.has_new_value_:
            out.putVarInt32(8)
            out.putVarUint64(self.new_value_)
        if self.has_increment_status_:
            out.putVarInt32(16)
            out.putVarInt32(self.increment_status_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_new_value(d.getVarUint64())
                continue
            if tt == 16:
                self.set_increment_status(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_new_value_:
            res += prefix + 'new_value: %s\n' % self.DebugFormatInt64(self.new_value_)
        if self.has_increment_status_:
            res += prefix + 'increment_status: %s\n' % self.DebugFormatInt32(self.increment_status_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    knew_value = 1
    kincrement_status = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'new_value', 2: 'increment_status'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC, 2: ProtocolBuffer.Encoder.NUMERIC}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheIncrementResponse'