from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class MemcacheGetResponse(ProtocolBuffer.ProtocolMessage):
    HIT = 1
    MISS = 2
    TRUNCATED = 3
    DEADLINE_EXCEEDED = 4
    UNREACHABLE = 5
    OTHER_ERROR = 6
    _GetStatusCode_NAMES = {1: 'HIT', 2: 'MISS', 3: 'TRUNCATED', 4: 'DEADLINE_EXCEEDED', 5: 'UNREACHABLE', 6: 'OTHER_ERROR'}

    def GetStatusCode_Name(cls, x):
        return cls._GetStatusCode_NAMES.get(x, '')
    GetStatusCode_Name = classmethod(GetStatusCode_Name)

    def __init__(self, contents=None):
        self.item_ = []
        self.get_status_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def item_size(self):
        return len(self.item_)

    def item_list(self):
        return self.item_

    def item(self, i):
        return self.item_[i]

    def mutable_item(self, i):
        return self.item_[i]

    def add_item(self):
        x = MemcacheGetResponse_Item()
        self.item_.append(x)
        return x

    def clear_item(self):
        self.item_ = []

    def get_status_size(self):
        return len(self.get_status_)

    def get_status_list(self):
        return self.get_status_

    def get_status(self, i):
        return self.get_status_[i]

    def set_get_status(self, i, x):
        self.get_status_[i] = x

    def add_get_status(self, x):
        self.get_status_.append(x)

    def clear_get_status(self):
        self.get_status_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.item_size()):
            self.add_item().CopyFrom(x.item(i))
        for i in range(x.get_status_size()):
            self.add_get_status(x.get_status(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.item_) != len(x.item_):
            return 0
        for e1, e2 in zip(self.item_, x.item_):
            if e1 != e2:
                return 0
        if len(self.get_status_) != len(x.get_status_):
            return 0
        for e1, e2 in zip(self.get_status_, x.get_status_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.item_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 2 * len(self.item_)
        for i in range(len(self.item_)):
            n += self.item_[i].ByteSize()
        n += 1 * len(self.get_status_)
        for i in range(len(self.get_status_)):
            n += self.lengthVarInt64(self.get_status_[i])
        return n

    def ByteSizePartial(self):
        n = 0
        n += 2 * len(self.item_)
        for i in range(len(self.item_)):
            n += self.item_[i].ByteSizePartial()
        n += 1 * len(self.get_status_)
        for i in range(len(self.get_status_)):
            n += self.lengthVarInt64(self.get_status_[i])
        return n

    def Clear(self):
        self.clear_item()
        self.clear_get_status()

    def OutputUnchecked(self, out):
        for i in range(len(self.item_)):
            out.putVarInt32(11)
            self.item_[i].OutputUnchecked(out)
            out.putVarInt32(12)
        for i in range(len(self.get_status_)):
            out.putVarInt32(56)
            out.putVarInt32(self.get_status_[i])

    def OutputPartial(self, out):
        for i in range(len(self.item_)):
            out.putVarInt32(11)
            self.item_[i].OutputPartial(out)
            out.putVarInt32(12)
        for i in range(len(self.get_status_)):
            out.putVarInt32(56)
            out.putVarInt32(self.get_status_[i])

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 11:
                self.add_item().TryMerge(d)
                continue
            if tt == 56:
                self.add_get_status(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.item_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'Item%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        cnt = 0
        for e in self.get_status_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'get_status%s: %s\n' % (elm, self.DebugFormatInt32(e))
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kItemGroup = 1
    kItemkey = 2
    kItemvalue = 3
    kItemflags = 4
    kItemcas_id = 5
    kItemexpires_in_seconds = 6
    kget_status = 7
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'Item', 2: 'key', 3: 'value', 4: 'flags', 5: 'cas_id', 6: 'expires_in_seconds', 7: 'get_status'}, 7)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STARTGROUP, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.STRING, 4: ProtocolBuffer.Encoder.FLOAT, 5: ProtocolBuffer.Encoder.DOUBLE, 6: ProtocolBuffer.Encoder.NUMERIC, 7: ProtocolBuffer.Encoder.NUMERIC}, 7, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.MemcacheGetResponse'