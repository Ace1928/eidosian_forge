from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheDeleteRequest_Item(ProtocolBuffer.ProtocolMessage):
    has_key_ = 0
    key_ = ''
    has_delete_time_ = 0
    delete_time_ = 0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def key(self):
        return self.key_

    def set_key(self, x):
        self.has_key_ = 1
        self.key_ = x

    def clear_key(self):
        if self.has_key_:
            self.has_key_ = 0
            self.key_ = ''

    def has_key(self):
        return self.has_key_

    def delete_time(self):
        return self.delete_time_

    def set_delete_time(self, x):
        self.has_delete_time_ = 1
        self.delete_time_ = x

    def clear_delete_time(self):
        if self.has_delete_time_:
            self.has_delete_time_ = 0
            self.delete_time_ = 0

    def has_delete_time(self):
        return self.has_delete_time_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_key():
            self.set_key(x.key())
        if x.has_delete_time():
            self.set_delete_time(x.delete_time())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_delete_time_ != x.has_delete_time_:
            return 0
        if self.has_delete_time_ and self.delete_time_ != x.delete_time_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_key_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: key not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.key_))
        if self.has_delete_time_:
            n += 5
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_key_:
            n += 1
            n += self.lengthString(len(self.key_))
        if self.has_delete_time_:
            n += 5
        return n

    def Clear(self):
        self.clear_key()
        self.clear_delete_time()

    def OutputUnchecked(self, out):
        out.putVarInt32(18)
        out.putPrefixedString(self.key_)
        if self.has_delete_time_:
            out.putVarInt32(29)
            out.put32(self.delete_time_)

    def OutputPartial(self, out):
        if self.has_key_:
            out.putVarInt32(18)
            out.putPrefixedString(self.key_)
        if self.has_delete_time_:
            out.putVarInt32(29)
            out.put32(self.delete_time_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 12:
                break
            if tt == 18:
                self.set_key(d.getPrefixedString())
                continue
            if tt == 29:
                self.set_delete_time(d.get32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_key_:
            res += prefix + 'key: %s\n' % self.DebugFormatString(self.key_)
        if self.has_delete_time_:
            res += prefix + 'delete_time: %s\n' % self.DebugFormatFixed32(self.delete_time_)
        return res