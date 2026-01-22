from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class CapabilityConfigList(ProtocolBuffer.ProtocolMessage):
    has_default_config_ = 0
    default_config_ = None

    def __init__(self, contents=None):
        self.config_ = []
        self.lazy_init_lock_ = _Lock()
        if contents is not None:
            self.MergeFromString(contents)

    def config_size(self):
        return len(self.config_)

    def config_list(self):
        return self.config_

    def config(self, i):
        return self.config_[i]

    def mutable_config(self, i):
        return self.config_[i]

    def add_config(self):
        x = CapabilityConfig()
        self.config_.append(x)
        return x

    def clear_config(self):
        self.config_ = []

    def default_config(self):
        if self.default_config_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.default_config_ is None:
                    self.default_config_ = CapabilityConfig()
            finally:
                self.lazy_init_lock_.release()
        return self.default_config_

    def mutable_default_config(self):
        self.has_default_config_ = 1
        return self.default_config()

    def clear_default_config(self):
        if self.has_default_config_:
            self.has_default_config_ = 0
            if self.default_config_ is not None:
                self.default_config_.Clear()

    def has_default_config(self):
        return self.has_default_config_

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.config_size()):
            self.add_config().CopyFrom(x.config(i))
        if x.has_default_config():
            self.mutable_default_config().MergeFrom(x.default_config())

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.config_) != len(x.config_):
            return 0
        for e1, e2 in zip(self.config_, x.config_):
            if e1 != e2:
                return 0
        if self.has_default_config_ != x.has_default_config_:
            return 0
        if self.has_default_config_ and self.default_config_ != x.default_config_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.config_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        if self.has_default_config_ and (not self.default_config_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.config_)
        for i in range(len(self.config_)):
            n += self.lengthString(self.config_[i].ByteSize())
        if self.has_default_config_:
            n += 1 + self.lengthString(self.default_config_.ByteSize())
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.config_)
        for i in range(len(self.config_)):
            n += self.lengthString(self.config_[i].ByteSizePartial())
        if self.has_default_config_:
            n += 1 + self.lengthString(self.default_config_.ByteSizePartial())
        return n

    def Clear(self):
        self.clear_config()
        self.clear_default_config()

    def OutputUnchecked(self, out):
        for i in range(len(self.config_)):
            out.putVarInt32(10)
            out.putVarInt32(self.config_[i].ByteSize())
            self.config_[i].OutputUnchecked(out)
        if self.has_default_config_:
            out.putVarInt32(18)
            out.putVarInt32(self.default_config_.ByteSize())
            self.default_config_.OutputUnchecked(out)

    def OutputPartial(self, out):
        for i in range(len(self.config_)):
            out.putVarInt32(10)
            out.putVarInt32(self.config_[i].ByteSizePartial())
            self.config_[i].OutputPartial(out)
        if self.has_default_config_:
            out.putVarInt32(18)
            out.putVarInt32(self.default_config_.ByteSizePartial())
            self.default_config_.OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_config().TryMerge(tmp)
                continue
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_default_config().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.config_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'config%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        if self.has_default_config_:
            res += prefix + 'default_config <\n'
            res += self.default_config_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kconfig = 1
    kdefault_config = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'config', 2: 'default_config'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.CapabilityConfigList'