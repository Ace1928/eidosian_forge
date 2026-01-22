from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class EntityMetadata(ProtocolBuffer.ProtocolMessage):
    has_created_version_ = 0
    created_version_ = 0
    has_updated_version_ = 0
    updated_version_ = 0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def created_version(self):
        return self.created_version_

    def set_created_version(self, x):
        self.has_created_version_ = 1
        self.created_version_ = x

    def clear_created_version(self):
        if self.has_created_version_:
            self.has_created_version_ = 0
            self.created_version_ = 0

    def has_created_version(self):
        return self.has_created_version_

    def updated_version(self):
        return self.updated_version_

    def set_updated_version(self, x):
        self.has_updated_version_ = 1
        self.updated_version_ = x

    def clear_updated_version(self):
        if self.has_updated_version_:
            self.has_updated_version_ = 0
            self.updated_version_ = 0

    def has_updated_version(self):
        return self.has_updated_version_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_created_version():
            self.set_created_version(x.created_version())
        if x.has_updated_version():
            self.set_updated_version(x.updated_version())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_created_version_ != x.has_created_version_:
            return 0
        if self.has_created_version_ and self.created_version_ != x.created_version_:
            return 0
        if self.has_updated_version_ != x.has_updated_version_:
            return 0
        if self.has_updated_version_ and self.updated_version_ != x.updated_version_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_created_version_:
            n += 1 + self.lengthVarInt64(self.created_version_)
        if self.has_updated_version_:
            n += 1 + self.lengthVarInt64(self.updated_version_)
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_created_version_:
            n += 1 + self.lengthVarInt64(self.created_version_)
        if self.has_updated_version_:
            n += 1 + self.lengthVarInt64(self.updated_version_)
        return n

    def Clear(self):
        self.clear_created_version()
        self.clear_updated_version()

    def OutputUnchecked(self, out):
        if self.has_created_version_:
            out.putVarInt32(8)
            out.putVarInt64(self.created_version_)
        if self.has_updated_version_:
            out.putVarInt32(16)
            out.putVarInt64(self.updated_version_)

    def OutputPartial(self, out):
        if self.has_created_version_:
            out.putVarInt32(8)
            out.putVarInt64(self.created_version_)
        if self.has_updated_version_:
            out.putVarInt32(16)
            out.putVarInt64(self.updated_version_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_created_version(d.getVarInt64())
                continue
            if tt == 16:
                self.set_updated_version(d.getVarInt64())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_created_version_:
            res += prefix + 'created_version: %s\n' % self.DebugFormatInt64(self.created_version_)
        if self.has_updated_version_:
            res += prefix + 'updated_version: %s\n' % self.DebugFormatInt64(self.updated_version_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kcreated_version = 1
    kupdated_version = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'created_version', 2: 'updated_version'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC, 2: ProtocolBuffer.Encoder.NUMERIC}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'storage_onestore_v3.EntityMetadata'