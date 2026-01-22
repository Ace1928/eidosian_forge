from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
class PropertyExpression(ProtocolBuffer.ProtocolMessage):
    FIRST = 1
    _AggregationFunction_NAMES = {1: 'FIRST'}

    def AggregationFunction_Name(cls, x):
        return cls._AggregationFunction_NAMES.get(x, '')
    AggregationFunction_Name = classmethod(AggregationFunction_Name)
    has_property_ = 0
    has_aggregation_function_ = 0
    aggregation_function_ = 0

    def __init__(self, contents=None):
        self.property_ = PropertyReference()
        if contents is not None:
            self.MergeFromString(contents)

    def property(self):
        return self.property_

    def mutable_property(self):
        self.has_property_ = 1
        return self.property_

    def clear_property(self):
        self.has_property_ = 0
        self.property_.Clear()

    def has_property(self):
        return self.has_property_

    def aggregation_function(self):
        return self.aggregation_function_

    def set_aggregation_function(self, x):
        self.has_aggregation_function_ = 1
        self.aggregation_function_ = x

    def clear_aggregation_function(self):
        if self.has_aggregation_function_:
            self.has_aggregation_function_ = 0
            self.aggregation_function_ = 0

    def has_aggregation_function(self):
        return self.has_aggregation_function_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_property():
            self.mutable_property().MergeFrom(x.property())
        if x.has_aggregation_function():
            self.set_aggregation_function(x.aggregation_function())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_property_ != x.has_property_:
            return 0
        if self.has_property_ and self.property_ != x.property_:
            return 0
        if self.has_aggregation_function_ != x.has_aggregation_function_:
            return 0
        if self.has_aggregation_function_ and self.aggregation_function_ != x.aggregation_function_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_property_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: property not set.')
        elif not self.property_.IsInitialized(debug_strs):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(self.property_.ByteSize())
        if self.has_aggregation_function_:
            n += 1 + self.lengthVarInt64(self.aggregation_function_)
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_property_:
            n += 1
            n += self.lengthString(self.property_.ByteSizePartial())
        if self.has_aggregation_function_:
            n += 1 + self.lengthVarInt64(self.aggregation_function_)
        return n

    def Clear(self):
        self.clear_property()
        self.clear_aggregation_function()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putVarInt32(self.property_.ByteSize())
        self.property_.OutputUnchecked(out)
        if self.has_aggregation_function_:
            out.putVarInt32(16)
            out.putVarInt32(self.aggregation_function_)

    def OutputPartial(self, out):
        if self.has_property_:
            out.putVarInt32(10)
            out.putVarInt32(self.property_.ByteSizePartial())
            self.property_.OutputPartial(out)
        if self.has_aggregation_function_:
            out.putVarInt32(16)
            out.putVarInt32(self.aggregation_function_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_property().TryMerge(tmp)
                continue
            if tt == 16:
                self.set_aggregation_function(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_property_:
            res += prefix + 'property <\n'
            res += self.property_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_aggregation_function_:
            res += prefix + 'aggregation_function: %s\n' % self.DebugFormatInt32(self.aggregation_function_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kproperty = 1
    kaggregation_function = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'property', 2: 'aggregation_function'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.NUMERIC}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.datastore.v4.PropertyExpression'