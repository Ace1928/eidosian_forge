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
class AddActionsResponse(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        pass
        if contents is not None:
            self.MergeFromString(contents)

    def MergeFrom(self, x):
        assert x is not self

    def Equals(self, x):
        if x is self:
            return 1
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        return n

    def ByteSizePartial(self):
        n = 0
        return n

    def Clear(self):
        pass

    def OutputUnchecked(self, out):
        pass

    def OutputPartial(self, out):
        pass

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode'}, 0)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC}, 0, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.AddActionsResponse'