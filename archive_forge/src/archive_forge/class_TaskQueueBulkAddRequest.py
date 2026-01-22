from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueBulkAddRequest(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        self.add_request_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def add_request_size(self):
        return len(self.add_request_)

    def add_request_list(self):
        return self.add_request_

    def add_request(self, i):
        return self.add_request_[i]

    def mutable_add_request(self, i):
        return self.add_request_[i]

    def add_add_request(self):
        x = TaskQueueAddRequest()
        self.add_request_.append(x)
        return x

    def clear_add_request(self):
        self.add_request_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.add_request_size()):
            self.add_add_request().CopyFrom(x.add_request(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.add_request_) != len(x.add_request_):
            return 0
        for e1, e2 in zip(self.add_request_, x.add_request_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.add_request_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.add_request_)
        for i in range(len(self.add_request_)):
            n += self.lengthString(self.add_request_[i].ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.add_request_)
        for i in range(len(self.add_request_)):
            n += self.lengthString(self.add_request_[i].ByteSizePartial())
        return n

    def Clear(self):
        self.clear_add_request()

    def OutputUnchecked(self, out):
        for i in range(len(self.add_request_)):
            out.putVarInt32(10)
            out.putVarInt32(self.add_request_[i].ByteSize())
            self.add_request_[i].OutputUnchecked(out)

    def OutputPartial(self, out):
        for i in range(len(self.add_request_)):
            out.putVarInt32(10)
            out.putVarInt32(self.add_request_[i].ByteSizePartial())
            self.add_request_[i].OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_add_request().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.add_request_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'add_request%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kadd_request = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'add_request'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueBulkAddRequest'