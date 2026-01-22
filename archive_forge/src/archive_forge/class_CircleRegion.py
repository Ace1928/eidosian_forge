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
class CircleRegion(ProtocolBuffer.ProtocolMessage):
    has_center_ = 0
    has_radius_meters_ = 0
    radius_meters_ = 0.0

    def __init__(self, contents=None):
        self.center_ = RegionPoint()
        if contents is not None:
            self.MergeFromString(contents)

    def center(self):
        return self.center_

    def mutable_center(self):
        self.has_center_ = 1
        return self.center_

    def clear_center(self):
        self.has_center_ = 0
        self.center_.Clear()

    def has_center(self):
        return self.has_center_

    def radius_meters(self):
        return self.radius_meters_

    def set_radius_meters(self, x):
        self.has_radius_meters_ = 1
        self.radius_meters_ = x

    def clear_radius_meters(self):
        if self.has_radius_meters_:
            self.has_radius_meters_ = 0
            self.radius_meters_ = 0.0

    def has_radius_meters(self):
        return self.has_radius_meters_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_center():
            self.mutable_center().MergeFrom(x.center())
        if x.has_radius_meters():
            self.set_radius_meters(x.radius_meters())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_center_ != x.has_center_:
            return 0
        if self.has_center_ and self.center_ != x.center_:
            return 0
        if self.has_radius_meters_ != x.has_radius_meters_:
            return 0
        if self.has_radius_meters_ and self.radius_meters_ != x.radius_meters_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_center_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: center not set.')
        elif not self.center_.IsInitialized(debug_strs):
            initialized = 0
        if not self.has_radius_meters_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: radius_meters not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(self.center_.ByteSize())
        return n + 10

    def ByteSizePartial(self):
        n = 0
        if self.has_center_:
            n += 1
            n += self.lengthString(self.center_.ByteSizePartial())
        if self.has_radius_meters_:
            n += 9
        return n

    def Clear(self):
        self.clear_center()
        self.clear_radius_meters()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putVarInt32(self.center_.ByteSize())
        self.center_.OutputUnchecked(out)
        out.putVarInt32(17)
        out.putDouble(self.radius_meters_)

    def OutputPartial(self, out):
        if self.has_center_:
            out.putVarInt32(10)
            out.putVarInt32(self.center_.ByteSizePartial())
            self.center_.OutputPartial(out)
        if self.has_radius_meters_:
            out.putVarInt32(17)
            out.putDouble(self.radius_meters_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_center().TryMerge(tmp)
                continue
            if tt == 17:
                self.set_radius_meters(d.getDouble())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_center_:
            res += prefix + 'center <\n'
            res += self.center_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_radius_meters_:
            res += prefix + 'radius_meters: %s\n' % self.DebugFormat(self.radius_meters_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kcenter = 1
    kradius_meters = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'center', 2: 'radius_meters'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.DOUBLE}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.CircleRegion'