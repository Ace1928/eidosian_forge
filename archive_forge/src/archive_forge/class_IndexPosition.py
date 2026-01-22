from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class IndexPosition(ProtocolBuffer.ProtocolMessage):
    has_key_ = 0
    key_ = ''
    has_before_ = 0
    before_ = 1
    has_before_ascending_ = 0
    before_ascending_ = 0

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

    def before(self):
        return self.before_

    def set_before(self, x):
        self.has_before_ = 1
        self.before_ = x

    def clear_before(self):
        if self.has_before_:
            self.has_before_ = 0
            self.before_ = 1

    def has_before(self):
        return self.has_before_

    def before_ascending(self):
        return self.before_ascending_

    def set_before_ascending(self, x):
        self.has_before_ascending_ = 1
        self.before_ascending_ = x

    def clear_before_ascending(self):
        if self.has_before_ascending_:
            self.has_before_ascending_ = 0
            self.before_ascending_ = 0

    def has_before_ascending(self):
        return self.has_before_ascending_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_key():
            self.set_key(x.key())
        if x.has_before():
            self.set_before(x.before())
        if x.has_before_ascending():
            self.set_before_ascending(x.before_ascending())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_key_ != x.has_key_:
            return 0
        if self.has_key_ and self.key_ != x.key_:
            return 0
        if self.has_before_ != x.has_before_:
            return 0
        if self.has_before_ and self.before_ != x.before_:
            return 0
        if self.has_before_ascending_ != x.has_before_ascending_:
            return 0
        if self.has_before_ascending_ and self.before_ascending_ != x.before_ascending_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_key_:
            n += 1 + self.lengthString(len(self.key_))
        if self.has_before_:
            n += 2
        if self.has_before_ascending_:
            n += 2
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_key_:
            n += 1 + self.lengthString(len(self.key_))
        if self.has_before_:
            n += 2
        if self.has_before_ascending_:
            n += 2
        return n

    def Clear(self):
        self.clear_key()
        self.clear_before()
        self.clear_before_ascending()

    def OutputUnchecked(self, out):
        if self.has_key_:
            out.putVarInt32(10)
            out.putPrefixedString(self.key_)
        if self.has_before_:
            out.putVarInt32(16)
            out.putBoolean(self.before_)
        if self.has_before_ascending_:
            out.putVarInt32(24)
            out.putBoolean(self.before_ascending_)

    def OutputPartial(self, out):
        if self.has_key_:
            out.putVarInt32(10)
            out.putPrefixedString(self.key_)
        if self.has_before_:
            out.putVarInt32(16)
            out.putBoolean(self.before_)
        if self.has_before_ascending_:
            out.putVarInt32(24)
            out.putBoolean(self.before_ascending_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_key(d.getPrefixedString())
                continue
            if tt == 16:
                self.set_before(d.getBoolean())
                continue
            if tt == 24:
                self.set_before_ascending(d.getBoolean())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_key_:
            res += prefix + 'key: %s\n' % self.DebugFormatString(self.key_)
        if self.has_before_:
            res += prefix + 'before: %s\n' % self.DebugFormatBool(self.before_)
        if self.has_before_ascending_:
            res += prefix + 'before_ascending: %s\n' % self.DebugFormatBool(self.before_ascending_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kkey = 1
    kbefore = 2
    kbefore_ascending = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'key', 2: 'before', 3: 'before_ascending'}, 3)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.NUMERIC, 3: ProtocolBuffer.Encoder.NUMERIC}, 3, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.IndexPosition'