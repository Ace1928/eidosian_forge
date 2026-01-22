from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class EntitySummary_PropertySummary(ProtocolBuffer.ProtocolMessage):
    has_name_ = 0
    name_ = ''
    has_property_type_for_stats_ = 0
    property_type_for_stats_ = ''
    has_size_bytes_ = 0
    size_bytes_ = 0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def name(self):
        return self.name_

    def set_name(self, x):
        self.has_name_ = 1
        self.name_ = x

    def clear_name(self):
        if self.has_name_:
            self.has_name_ = 0
            self.name_ = ''

    def has_name(self):
        return self.has_name_

    def property_type_for_stats(self):
        return self.property_type_for_stats_

    def set_property_type_for_stats(self, x):
        self.has_property_type_for_stats_ = 1
        self.property_type_for_stats_ = x

    def clear_property_type_for_stats(self):
        if self.has_property_type_for_stats_:
            self.has_property_type_for_stats_ = 0
            self.property_type_for_stats_ = ''

    def has_property_type_for_stats(self):
        return self.has_property_type_for_stats_

    def size_bytes(self):
        return self.size_bytes_

    def set_size_bytes(self, x):
        self.has_size_bytes_ = 1
        self.size_bytes_ = x

    def clear_size_bytes(self):
        if self.has_size_bytes_:
            self.has_size_bytes_ = 0
            self.size_bytes_ = 0

    def has_size_bytes(self):
        return self.has_size_bytes_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_name():
            self.set_name(x.name())
        if x.has_property_type_for_stats():
            self.set_property_type_for_stats(x.property_type_for_stats())
        if x.has_size_bytes():
            self.set_size_bytes(x.size_bytes())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_name_ != x.has_name_:
            return 0
        if self.has_name_ and self.name_ != x.name_:
            return 0
        if self.has_property_type_for_stats_ != x.has_property_type_for_stats_:
            return 0
        if self.has_property_type_for_stats_ and self.property_type_for_stats_ != x.property_type_for_stats_:
            return 0
        if self.has_size_bytes_ != x.has_size_bytes_:
            return 0
        if self.has_size_bytes_ and self.size_bytes_ != x.size_bytes_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: name not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.name_))
        if self.has_property_type_for_stats_:
            n += 1 + self.lengthString(len(self.property_type_for_stats_))
        if self.has_size_bytes_:
            n += 1 + self.lengthVarInt64(self.size_bytes_)
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_name_:
            n += 1
            n += self.lengthString(len(self.name_))
        if self.has_property_type_for_stats_:
            n += 1 + self.lengthString(len(self.property_type_for_stats_))
        if self.has_size_bytes_:
            n += 1 + self.lengthVarInt64(self.size_bytes_)
        return n

    def Clear(self):
        self.clear_name()
        self.clear_property_type_for_stats()
        self.clear_size_bytes()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.name_)
        if self.has_property_type_for_stats_:
            out.putVarInt32(18)
            out.putPrefixedString(self.property_type_for_stats_)
        if self.has_size_bytes_:
            out.putVarInt32(24)
            out.putVarInt32(self.size_bytes_)

    def OutputPartial(self, out):
        if self.has_name_:
            out.putVarInt32(10)
            out.putPrefixedString(self.name_)
        if self.has_property_type_for_stats_:
            out.putVarInt32(18)
            out.putPrefixedString(self.property_type_for_stats_)
        if self.has_size_bytes_:
            out.putVarInt32(24)
            out.putVarInt32(self.size_bytes_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_name(d.getPrefixedString())
                continue
            if tt == 18:
                self.set_property_type_for_stats(d.getPrefixedString())
                continue
            if tt == 24:
                self.set_size_bytes(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_name_:
            res += prefix + 'name: %s\n' % self.DebugFormatString(self.name_)
        if self.has_property_type_for_stats_:
            res += prefix + 'property_type_for_stats: %s\n' % self.DebugFormatString(self.property_type_for_stats_)
        if self.has_size_bytes_:
            res += prefix + 'size_bytes: %s\n' % self.DebugFormatInt32(self.size_bytes_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kname = 1
    kproperty_type_for_stats = 2
    ksize_bytes = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'name', 2: 'property_type_for_stats', 3: 'size_bytes'}, 3)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC}, 3, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.EntitySummary_PropertySummary'