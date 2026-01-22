from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheGetResponse_Item(ProtocolBuffer.ProtocolMessage):
    has_key_ = 0
    key_ = ''
    has_value_ = 0
    value_ = ''
    has_flags_ = 0
    flags_ = 0
    has_cas_id_ = 0
    cas_id_ = 0
    has_expires_in_seconds_ = 0
    expires_in_seconds_ = 0

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

    def value(self):
        return self.value_

    def set_value(self, x):
        self.has_value_ = 1
        self.value_ = x

    def clear_value(self):
        if self.has_value_:
            self.has_value_ = 0
            self.value_ = ''

    def has_value(self):
        return self.has_value_

    def flags(self):
        return self.flags_

    def set_flags(self, x):
        self.has_flags_ = 1
        self.flags_ = x

    def clear_flags(self):
        if self.has_flags_:
            self.has_flags_ = 0
            self.flags_ = 0

    def has_flags(self):
        return self.has_flags_

    def cas_id(self):
        return self.cas_id_

    def set_cas_id(self, x):
        self.has_cas_id_ = 1
        self.cas_id_ = x

    def clear_cas_id(self):
        if self.has_cas_id_:
            self.has_cas_id_ = 0
            self.cas_id_ = 0

    def has_cas_id(self):
        return self.has_cas_id_

    def expires_in_seconds(self):
        return self.expires_in_seconds_

    def set_expires_in_seconds(self, x):
        self.has_expires_in_seconds_ = 1
        self.expires_in_seconds_ = x

    def clear_expires_in_seconds(self):
        if self.has_expires_in_seconds_:
            self.has_expires_in_seconds_ = 0
            self.expires_in_seconds_ = 0

    def has_expires_in_seconds(self):
        return self.has_expires_in_seconds_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_key():
            self.set_key(x.key())
        if x.has_value():
            self.set_value(x.value())
        if x.has_flags():
            self.set_flags(x.flags())
        if x.has_cas_id():
            self.set_cas_id(x.cas_id())
        if x.has_expires_in_seconds():
            self.set_expires_in_seconds(x.expires_in_seconds())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_value_ != x.has_value_:
            return 0
        if self.has_value_ and self.value_ != x.value_:
            return 0
        if self.has_flags_ != x.has_flags_:
            return 0
        if self.has_flags_ and self.flags_ != x.flags_:
            return 0
        if self.has_cas_id_ != x.has_cas_id_:
            return 0
        if self.has_cas_id_ and self.cas_id_ != x.cas_id_:
            return 0
        if self.has_expires_in_seconds_ != x.has_expires_in_seconds_:
            return 0
        if self.has_expires_in_seconds_ and self.expires_in_seconds_ != x.expires_in_seconds_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_key_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: key not set.')
        if not self.has_value_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: value not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.key_))
        n += self.lengthString(len(self.value_))
        if self.has_flags_:
            n += 5
        if self.has_cas_id_:
            n += 9
        if self.has_expires_in_seconds_:
            n += 1 + self.lengthVarInt64(self.expires_in_seconds_)
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_key_:
            n += 1
            n += self.lengthString(len(self.key_))
        if self.has_value_:
            n += 1
            n += self.lengthString(len(self.value_))
        if self.has_flags_:
            n += 5
        if self.has_cas_id_:
            n += 9
        if self.has_expires_in_seconds_:
            n += 1 + self.lengthVarInt64(self.expires_in_seconds_)
        return n

    def Clear(self):
        self.clear_key()
        self.clear_value()
        self.clear_flags()
        self.clear_cas_id()
        self.clear_expires_in_seconds()

    def OutputUnchecked(self, out):
        out.putVarInt32(18)
        out.putPrefixedString(self.key_)
        out.putVarInt32(26)
        out.putPrefixedString(self.value_)
        if self.has_flags_:
            out.putVarInt32(37)
            out.put32(self.flags_)
        if self.has_cas_id_:
            out.putVarInt32(41)
            out.put64(self.cas_id_)
        if self.has_expires_in_seconds_:
            out.putVarInt32(48)
            out.putVarInt32(self.expires_in_seconds_)

    def OutputPartial(self, out):
        if self.has_key_:
            out.putVarInt32(18)
            out.putPrefixedString(self.key_)
        if self.has_value_:
            out.putVarInt32(26)
            out.putPrefixedString(self.value_)
        if self.has_flags_:
            out.putVarInt32(37)
            out.put32(self.flags_)
        if self.has_cas_id_:
            out.putVarInt32(41)
            out.put64(self.cas_id_)
        if self.has_expires_in_seconds_:
            out.putVarInt32(48)
            out.putVarInt32(self.expires_in_seconds_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 12:
                break
            if tt == 18:
                self.set_key(d.getPrefixedString())
                continue
            if tt == 26:
                self.set_value(d.getPrefixedString())
                continue
            if tt == 37:
                self.set_flags(d.get32())
                continue
            if tt == 41:
                self.set_cas_id(d.get64())
                continue
            if tt == 48:
                self.set_expires_in_seconds(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_key_:
            res += prefix + 'key: %s\n' % self.DebugFormatString(self.key_)
        if self.has_value_:
            res += prefix + 'value: %s\n' % self.DebugFormatString(self.value_)
        if self.has_flags_:
            res += prefix + 'flags: %s\n' % self.DebugFormatFixed32(self.flags_)
        if self.has_cas_id_:
            res += prefix + 'cas_id: %s\n' % self.DebugFormatFixed64(self.cas_id_)
        if self.has_expires_in_seconds_:
            res += prefix + 'expires_in_seconds: %s\n' % self.DebugFormatInt32(self.expires_in_seconds_)
        return res