from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class EntitySummary(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        self.large_raw_property_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def large_raw_property_size(self):
        return len(self.large_raw_property_)

    def large_raw_property_list(self):
        return self.large_raw_property_

    def large_raw_property(self, i):
        return self.large_raw_property_[i]

    def mutable_large_raw_property(self, i):
        return self.large_raw_property_[i]

    def add_large_raw_property(self):
        x = EntitySummary_PropertySummary()
        self.large_raw_property_.append(x)
        return x

    def clear_large_raw_property(self):
        self.large_raw_property_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.large_raw_property_size()):
            self.add_large_raw_property().CopyFrom(x.large_raw_property(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.large_raw_property_) != len(x.large_raw_property_):
            return 0
        for e1, e2 in zip(self.large_raw_property_, x.large_raw_property_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.large_raw_property_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.large_raw_property_)
        for i in range(len(self.large_raw_property_)):
            n += self.lengthString(self.large_raw_property_[i].ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.large_raw_property_)
        for i in range(len(self.large_raw_property_)):
            n += self.lengthString(self.large_raw_property_[i].ByteSizePartial())
        return n

    def Clear(self):
        self.clear_large_raw_property()

    def OutputUnchecked(self, out):
        for i in range(len(self.large_raw_property_)):
            out.putVarInt32(10)
            out.putVarInt32(self.large_raw_property_[i].ByteSize())
            self.large_raw_property_[i].OutputUnchecked(out)

    def OutputPartial(self, out):
        for i in range(len(self.large_raw_property_)):
            out.putVarInt32(10)
            out.putVarInt32(self.large_raw_property_[i].ByteSizePartial())
            self.large_raw_property_[i].OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_large_raw_property().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.large_raw_property_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'large_raw_property%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    klarge_raw_property = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'large_raw_property'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.EntitySummary'