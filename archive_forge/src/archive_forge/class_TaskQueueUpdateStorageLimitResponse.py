from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueUpdateStorageLimitResponse(ProtocolBuffer.ProtocolMessage):
    has_new_limit_ = 0
    new_limit_ = 0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def new_limit(self):
        return self.new_limit_

    def set_new_limit(self, x):
        self.has_new_limit_ = 1
        self.new_limit_ = x

    def clear_new_limit(self):
        if self.has_new_limit_:
            self.has_new_limit_ = 0
            self.new_limit_ = 0

    def has_new_limit(self):
        return self.has_new_limit_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_new_limit():
            self.set_new_limit(x.new_limit())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_new_limit_ != x.has_new_limit_:
            return 0
        if self.has_new_limit_ and self.new_limit_ != x.new_limit_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_new_limit_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: new_limit not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthVarInt64(self.new_limit_)
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_new_limit_:
            n += 1
            n += self.lengthVarInt64(self.new_limit_)
        return n

    def Clear(self):
        self.clear_new_limit()

    def OutputUnchecked(self, out):
        out.putVarInt32(8)
        out.putVarInt64(self.new_limit_)

    def OutputPartial(self, out):
        if self.has_new_limit_:
            out.putVarInt32(8)
            out.putVarInt64(self.new_limit_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_new_limit(d.getVarInt64())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_new_limit_:
            res += prefix + 'new_limit: %s\n' % self.DebugFormatInt64(self.new_limit_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    knew_limit = 1
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'new_limit'}, 1)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC}, 1, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueUpdateStorageLimitResponse'