from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb import *
import googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.snapshot_pb
class CompositeIndices(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        self.index_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def index_size(self):
        return len(self.index_)

    def index_list(self):
        return self.index_

    def index(self, i):
        return self.index_[i]

    def mutable_index(self, i):
        return self.index_[i]

    def add_index(self):
        x = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.entity_pb.CompositeIndex()
        self.index_.append(x)
        return x

    def clear_index(self):
        self.index_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.index_size()):
            self.add_index().CopyFrom(x.index(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.index_) != len(x.index_):
            return 0
        for e1, e2 in zip(self.index_, x.index_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.index_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.index_)
        for i in range(len(self.index_)):
            n += self.lengthString(self.index_[i].ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.index_)
        for i in range(len(self.index_)):
            n += self.lengthString(self.index_[i].ByteSizePartial())
        return n

    def Clear(self):
        self.clear_index()

    def OutputUnchecked(self, out):
        for i in range(len(self.index_)):
            out.putVarInt32(10)
            out.putVarInt32(self.index_[i].ByteSize())
            self.index_[i].OutputUnchecked(out)

    def OutputPartial(self, out):
        for i in range(len(self.index_)):
            out.putVarInt32(10)
            out.putVarInt32(self.index_[i].ByteSizePartial())
            self.index_[i].OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_index().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.index_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'index%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kindex = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'index'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.CompositeIndices'