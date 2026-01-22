from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
class DeprecatedMutationResult(ProtocolBuffer.ProtocolMessage):
    has_index_updates_ = 0
    index_updates_ = 0

    def __init__(self, contents=None):
        self.insert_auto_id_key_ = []
        self.upsert_version_ = []
        self.update_version_ = []
        self.insert_version_ = []
        self.insert_auto_id_version_ = []
        self.delete_version_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def index_updates(self):
        return self.index_updates_

    def set_index_updates(self, x):
        self.has_index_updates_ = 1
        self.index_updates_ = x

    def clear_index_updates(self):
        if self.has_index_updates_:
            self.has_index_updates_ = 0
            self.index_updates_ = 0

    def has_index_updates(self):
        return self.has_index_updates_

    def insert_auto_id_key_size(self):
        return len(self.insert_auto_id_key_)

    def insert_auto_id_key_list(self):
        return self.insert_auto_id_key_

    def insert_auto_id_key(self, i):
        return self.insert_auto_id_key_[i]

    def mutable_insert_auto_id_key(self, i):
        return self.insert_auto_id_key_[i]

    def add_insert_auto_id_key(self):
        x = googlecloudsdk.third_party.appengine.datastore.entity_v4_pb.Key()
        self.insert_auto_id_key_.append(x)
        return x

    def clear_insert_auto_id_key(self):
        self.insert_auto_id_key_ = []

    def upsert_version_size(self):
        return len(self.upsert_version_)

    def upsert_version_list(self):
        return self.upsert_version_

    def upsert_version(self, i):
        return self.upsert_version_[i]

    def set_upsert_version(self, i, x):
        self.upsert_version_[i] = x

    def add_upsert_version(self, x):
        self.upsert_version_.append(x)

    def clear_upsert_version(self):
        self.upsert_version_ = []

    def update_version_size(self):
        return len(self.update_version_)

    def update_version_list(self):
        return self.update_version_

    def update_version(self, i):
        return self.update_version_[i]

    def set_update_version(self, i, x):
        self.update_version_[i] = x

    def add_update_version(self, x):
        self.update_version_.append(x)

    def clear_update_version(self):
        self.update_version_ = []

    def insert_version_size(self):
        return len(self.insert_version_)

    def insert_version_list(self):
        return self.insert_version_

    def insert_version(self, i):
        return self.insert_version_[i]

    def set_insert_version(self, i, x):
        self.insert_version_[i] = x

    def add_insert_version(self, x):
        self.insert_version_.append(x)

    def clear_insert_version(self):
        self.insert_version_ = []

    def insert_auto_id_version_size(self):
        return len(self.insert_auto_id_version_)

    def insert_auto_id_version_list(self):
        return self.insert_auto_id_version_

    def insert_auto_id_version(self, i):
        return self.insert_auto_id_version_[i]

    def set_insert_auto_id_version(self, i, x):
        self.insert_auto_id_version_[i] = x

    def add_insert_auto_id_version(self, x):
        self.insert_auto_id_version_.append(x)

    def clear_insert_auto_id_version(self):
        self.insert_auto_id_version_ = []

    def delete_version_size(self):
        return len(self.delete_version_)

    def delete_version_list(self):
        return self.delete_version_

    def delete_version(self, i):
        return self.delete_version_[i]

    def set_delete_version(self, i, x):
        self.delete_version_[i] = x

    def add_delete_version(self, x):
        self.delete_version_.append(x)

    def clear_delete_version(self):
        self.delete_version_ = []

    def MergeFrom(self, x):
        assert x is not self
        if x.has_index_updates():
            self.set_index_updates(x.index_updates())
        for i in range(x.insert_auto_id_key_size()):
            self.add_insert_auto_id_key().CopyFrom(x.insert_auto_id_key(i))
        for i in range(x.upsert_version_size()):
            self.add_upsert_version(x.upsert_version(i))
        for i in range(x.update_version_size()):
            self.add_update_version(x.update_version(i))
        for i in range(x.insert_version_size()):
            self.add_insert_version(x.insert_version(i))
        for i in range(x.insert_auto_id_version_size()):
            self.add_insert_auto_id_version(x.insert_auto_id_version(i))
        for i in range(x.delete_version_size()):
            self.add_delete_version(x.delete_version(i))

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_index_updates_ != x.has_index_updates_:
            return 0
        if self.has_index_updates_ and self.index_updates_ != x.index_updates_:
            return 0
        if len(self.insert_auto_id_key_) != len(x.insert_auto_id_key_):
            return 0
        for e1, e2 in zip(self.insert_auto_id_key_, x.insert_auto_id_key_):
            if e1 != e2:
                return 0
        if len(self.upsert_version_) != len(x.upsert_version_):
            return 0
        for e1, e2 in zip(self.upsert_version_, x.upsert_version_):
            if e1 != e2:
                return 0
        if len(self.update_version_) != len(x.update_version_):
            return 0
        for e1, e2 in zip(self.update_version_, x.update_version_):
            if e1 != e2:
                return 0
        if len(self.insert_version_) != len(x.insert_version_):
            return 0
        for e1, e2 in zip(self.insert_version_, x.insert_version_):
            if e1 != e2:
                return 0
        if len(self.insert_auto_id_version_) != len(x.insert_auto_id_version_):
            return 0
        for e1, e2 in zip(self.insert_auto_id_version_, x.insert_auto_id_version_):
            if e1 != e2:
                return 0
        if len(self.delete_version_) != len(x.delete_version_):
            return 0
        for e1, e2 in zip(self.delete_version_, x.delete_version_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_index_updates_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: index_updates not set.')
        for p in self.insert_auto_id_key_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthVarInt64(self.index_updates_)
        n += 1 * len(self.insert_auto_id_key_)
        for i in range(len(self.insert_auto_id_key_)):
            n += self.lengthString(self.insert_auto_id_key_[i].ByteSize())
        n += 1 * len(self.upsert_version_)
        for i in range(len(self.upsert_version_)):
            n += self.lengthVarInt64(self.upsert_version_[i])
        n += 1 * len(self.update_version_)
        for i in range(len(self.update_version_)):
            n += self.lengthVarInt64(self.update_version_[i])
        n += 1 * len(self.insert_version_)
        for i in range(len(self.insert_version_)):
            n += self.lengthVarInt64(self.insert_version_[i])
        n += 1 * len(self.insert_auto_id_version_)
        for i in range(len(self.insert_auto_id_version_)):
            n += self.lengthVarInt64(self.insert_auto_id_version_[i])
        n += 1 * len(self.delete_version_)
        for i in range(len(self.delete_version_)):
            n += self.lengthVarInt64(self.delete_version_[i])
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_index_updates_:
            n += 1
            n += self.lengthVarInt64(self.index_updates_)
        n += 1 * len(self.insert_auto_id_key_)
        for i in range(len(self.insert_auto_id_key_)):
            n += self.lengthString(self.insert_auto_id_key_[i].ByteSizePartial())
        n += 1 * len(self.upsert_version_)
        for i in range(len(self.upsert_version_)):
            n += self.lengthVarInt64(self.upsert_version_[i])
        n += 1 * len(self.update_version_)
        for i in range(len(self.update_version_)):
            n += self.lengthVarInt64(self.update_version_[i])
        n += 1 * len(self.insert_version_)
        for i in range(len(self.insert_version_)):
            n += self.lengthVarInt64(self.insert_version_[i])
        n += 1 * len(self.insert_auto_id_version_)
        for i in range(len(self.insert_auto_id_version_)):
            n += self.lengthVarInt64(self.insert_auto_id_version_[i])
        n += 1 * len(self.delete_version_)
        for i in range(len(self.delete_version_)):
            n += self.lengthVarInt64(self.delete_version_[i])
        return n

    def Clear(self):
        self.clear_index_updates()
        self.clear_insert_auto_id_key()
        self.clear_upsert_version()
        self.clear_update_version()
        self.clear_insert_version()
        self.clear_insert_auto_id_version()
        self.clear_delete_version()

    def OutputUnchecked(self, out):
        out.putVarInt32(8)
        out.putVarInt32(self.index_updates_)
        for i in range(len(self.insert_auto_id_key_)):
            out.putVarInt32(18)
            out.putVarInt32(self.insert_auto_id_key_[i].ByteSize())
            self.insert_auto_id_key_[i].OutputUnchecked(out)
        for i in range(len(self.upsert_version_)):
            out.putVarInt32(24)
            out.putVarInt64(self.upsert_version_[i])
        for i in range(len(self.update_version_)):
            out.putVarInt32(32)
            out.putVarInt64(self.update_version_[i])
        for i in range(len(self.insert_version_)):
            out.putVarInt32(40)
            out.putVarInt64(self.insert_version_[i])
        for i in range(len(self.insert_auto_id_version_)):
            out.putVarInt32(48)
            out.putVarInt64(self.insert_auto_id_version_[i])
        for i in range(len(self.delete_version_)):
            out.putVarInt32(56)
            out.putVarInt64(self.delete_version_[i])

    def OutputPartial(self, out):
        if self.has_index_updates_:
            out.putVarInt32(8)
            out.putVarInt32(self.index_updates_)
        for i in range(len(self.insert_auto_id_key_)):
            out.putVarInt32(18)
            out.putVarInt32(self.insert_auto_id_key_[i].ByteSizePartial())
            self.insert_auto_id_key_[i].OutputPartial(out)
        for i in range(len(self.upsert_version_)):
            out.putVarInt32(24)
            out.putVarInt64(self.upsert_version_[i])
        for i in range(len(self.update_version_)):
            out.putVarInt32(32)
            out.putVarInt64(self.update_version_[i])
        for i in range(len(self.insert_version_)):
            out.putVarInt32(40)
            out.putVarInt64(self.insert_version_[i])
        for i in range(len(self.insert_auto_id_version_)):
            out.putVarInt32(48)
            out.putVarInt64(self.insert_auto_id_version_[i])
        for i in range(len(self.delete_version_)):
            out.putVarInt32(56)
            out.putVarInt64(self.delete_version_[i])

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 8:
                self.set_index_updates(d.getVarInt32())
                continue
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_insert_auto_id_key().TryMerge(tmp)
                continue
            if tt == 24:
                self.add_upsert_version(d.getVarInt64())
                continue
            if tt == 32:
                self.add_update_version(d.getVarInt64())
                continue
            if tt == 40:
                self.add_insert_version(d.getVarInt64())
                continue
            if tt == 48:
                self.add_insert_auto_id_version(d.getVarInt64())
                continue
            if tt == 56:
                self.add_delete_version(d.getVarInt64())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_index_updates_:
            res += prefix + 'index_updates: %s\n' % self.DebugFormatInt32(self.index_updates_)
        cnt = 0
        for e in self.insert_auto_id_key_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'insert_auto_id_key%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        cnt = 0
        for e in self.upsert_version_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'upsert_version%s: %s\n' % (elm, self.DebugFormatInt64(e))
            cnt += 1
        cnt = 0
        for e in self.update_version_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'update_version%s: %s\n' % (elm, self.DebugFormatInt64(e))
            cnt += 1
        cnt = 0
        for e in self.insert_version_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'insert_version%s: %s\n' % (elm, self.DebugFormatInt64(e))
            cnt += 1
        cnt = 0
        for e in self.insert_auto_id_version_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'insert_auto_id_version%s: %s\n' % (elm, self.DebugFormatInt64(e))
            cnt += 1
        cnt = 0
        for e in self.delete_version_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'delete_version%s: %s\n' % (elm, self.DebugFormatInt64(e))
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kindex_updates = 1
    kinsert_auto_id_key = 2
    kupsert_version = 3
    kupdate_version = 4
    kinsert_version = 5
    kinsert_auto_id_version = 6
    kdelete_version = 7
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'index_updates', 2: 'insert_auto_id_key', 3: 'upsert_version', 4: 'update_version', 5: 'insert_version', 6: 'insert_auto_id_version', 7: 'delete_version'}, 7)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.NUMERIC, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.NUMERIC, 5: ProtocolBuffer.Encoder.NUMERIC, 6: ProtocolBuffer.Encoder.NUMERIC, 7: ProtocolBuffer.Encoder.NUMERIC}, 7, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.datastore.v4.DeprecatedMutationResult'