from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
class ContinueQueryResponse(ProtocolBuffer.ProtocolMessage):
    has_batch_ = 0

    def __init__(self, contents=None):
        self.batch_ = QueryResultBatch()
        if contents is not None:
            self.MergeFromString(contents)

    def batch(self):
        return self.batch_

    def mutable_batch(self):
        self.has_batch_ = 1
        return self.batch_

    def clear_batch(self):
        self.has_batch_ = 0
        self.batch_.Clear()

    def has_batch(self):
        return self.has_batch_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_batch():
            self.mutable_batch().MergeFrom(x.batch())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_batch_ != x.has_batch_:
            return 0
        if self.has_batch_ and self.batch_ != x.batch_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_batch_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: batch not set.')
        elif not self.batch_.IsInitialized(debug_strs):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(self.batch_.ByteSize())
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_batch_:
            n += 1
            n += self.lengthString(self.batch_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_batch()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putVarInt32(self.batch_.ByteSize())
        self.batch_.OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_batch_:
            out.putVarInt32(10)
            out.putVarInt32(self.batch_.ByteSizePartial())
            self.batch_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_batch().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_batch_:
            res += prefix + 'batch <\n'
            res += self.batch_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kbatch = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'batch'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.datastore.v4.ContinueQueryResponse'