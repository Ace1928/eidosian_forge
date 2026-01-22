from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
class DeprecatedMutation(ProtocolBuffer.ProtocolMessage):
    has_force_ = 0
    force_ = 0

    def __init__(self, contents=None):
        self.upsert_ = []
        self.update_ = []
        self.insert_ = []
        self.insert_auto_id_ = []
        self.delete_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def upsert_size(self):
        return len(self.upsert_)

    def upsert_list(self):
        return self.upsert_

    def upsert(self, i):
        return self.upsert_[i]

    def mutable_upsert(self, i):
        return self.upsert_[i]

    def add_upsert(self):
        x = googlecloudsdk.third_party.appengine.datastore.entity_v4_pb.Entity()
        self.upsert_.append(x)
        return x

    def clear_upsert(self):
        self.upsert_ = []

    def update_size(self):
        return len(self.update_)

    def update_list(self):
        return self.update_

    def update(self, i):
        return self.update_[i]

    def mutable_update(self, i):
        return self.update_[i]

    def add_update(self):
        x = googlecloudsdk.third_party.appengine.datastore.entity_v4_pb.Entity()
        self.update_.append(x)
        return x

    def clear_update(self):
        self.update_ = []

    def insert_size(self):
        return len(self.insert_)

    def insert_list(self):
        return self.insert_

    def insert(self, i):
        return self.insert_[i]

    def mutable_insert(self, i):
        return self.insert_[i]

    def add_insert(self):
        x = googlecloudsdk.third_party.appengine.datastore.entity_v4_pb.Entity()
        self.insert_.append(x)
        return x

    def clear_insert(self):
        self.insert_ = []

    def insert_auto_id_size(self):
        return len(self.insert_auto_id_)

    def insert_auto_id_list(self):
        return self.insert_auto_id_

    def insert_auto_id(self, i):
        return self.insert_auto_id_[i]

    def mutable_insert_auto_id(self, i):
        return self.insert_auto_id_[i]

    def add_insert_auto_id(self):
        x = googlecloudsdk.third_party.appengine.datastore.entity_v4_pb.Entity()
        self.insert_auto_id_.append(x)
        return x

    def clear_insert_auto_id(self):
        self.insert_auto_id_ = []

    def delete_size(self):
        return len(self.delete_)

    def delete_list(self):
        return self.delete_

    def delete(self, i):
        return self.delete_[i]

    def mutable_delete(self, i):
        return self.delete_[i]

    def add_delete(self):
        x = googlecloudsdk.third_party.appengine.datastore.entity_v4_pb.Key()
        self.delete_.append(x)
        return x

    def clear_delete(self):
        self.delete_ = []

    def force(self):
        return self.force_

    def set_force(self, x):
        self.has_force_ = 1
        self.force_ = x

    def clear_force(self):
        if self.has_force_:
            self.has_force_ = 0
            self.force_ = 0

    def has_force(self):
        return self.has_force_

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.upsert_size()):
            self.add_upsert().CopyFrom(x.upsert(i))
        for i in range(x.update_size()):
            self.add_update().CopyFrom(x.update(i))
        for i in range(x.insert_size()):
            self.add_insert().CopyFrom(x.insert(i))
        for i in range(x.insert_auto_id_size()):
            self.add_insert_auto_id().CopyFrom(x.insert_auto_id(i))
        for i in range(x.delete_size()):
            self.add_delete().CopyFrom(x.delete(i))
        if x.has_force():
            self.set_force(x.force())

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.upsert_) != len(x.upsert_):
            return 0
        for e1, e2 in zip(self.upsert_, x.upsert_):
            if e1 != e2:
                return 0
        if len(self.update_) != len(x.update_):
            return 0
        for e1, e2 in zip(self.update_, x.update_):
            if e1 != e2:
                return 0
        if len(self.insert_) != len(x.insert_):
            return 0
        for e1, e2 in zip(self.insert_, x.insert_):
            if e1 != e2:
                return 0
        if len(self.insert_auto_id_) != len(x.insert_auto_id_):
            return 0
        for e1, e2 in zip(self.insert_auto_id_, x.insert_auto_id_):
            if e1 != e2:
                return 0
        if len(self.delete_) != len(x.delete_):
            return 0
        for e1, e2 in zip(self.delete_, x.delete_):
            if e1 != e2:
                return 0
        if self.has_force_ != x.has_force_:
            return 0
        if self.has_force_ and self.force_ != x.force_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.upsert_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        for p in self.update_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        for p in self.insert_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        for p in self.insert_auto_id_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        for p in self.delete_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 1 * len(self.upsert_)
        for i in range(len(self.upsert_)):
            n += self.lengthString(self.upsert_[i].ByteSize())
        n += 1 * len(self.update_)
        for i in range(len(self.update_)):
            n += self.lengthString(self.update_[i].ByteSize())
        n += 1 * len(self.insert_)
        for i in range(len(self.insert_)):
            n += self.lengthString(self.insert_[i].ByteSize())
        n += 1 * len(self.insert_auto_id_)
        for i in range(len(self.insert_auto_id_)):
            n += self.lengthString(self.insert_auto_id_[i].ByteSize())
        n += 1 * len(self.delete_)
        for i in range(len(self.delete_)):
            n += self.lengthString(self.delete_[i].ByteSize())
        if self.has_force_:
            n += 2
        return n

    def ByteSizePartial(self):
        n = 0
        n += 1 * len(self.upsert_)
        for i in range(len(self.upsert_)):
            n += self.lengthString(self.upsert_[i].ByteSizePartial())
        n += 1 * len(self.update_)
        for i in range(len(self.update_)):
            n += self.lengthString(self.update_[i].ByteSizePartial())
        n += 1 * len(self.insert_)
        for i in range(len(self.insert_)):
            n += self.lengthString(self.insert_[i].ByteSizePartial())
        n += 1 * len(self.insert_auto_id_)
        for i in range(len(self.insert_auto_id_)):
            n += self.lengthString(self.insert_auto_id_[i].ByteSizePartial())
        n += 1 * len(self.delete_)
        for i in range(len(self.delete_)):
            n += self.lengthString(self.delete_[i].ByteSizePartial())
        if self.has_force_:
            n += 2
        return n

    def Clear(self):
        self.clear_upsert()
        self.clear_update()
        self.clear_insert()
        self.clear_insert_auto_id()
        self.clear_delete()
        self.clear_force()

    def OutputUnchecked(self, out):
        for i in range(len(self.upsert_)):
            out.putVarInt32(10)
            out.putVarInt32(self.upsert_[i].ByteSize())
            self.upsert_[i].OutputUnchecked(out)
        for i in range(len(self.update_)):
            out.putVarInt32(18)
            out.putVarInt32(self.update_[i].ByteSize())
            self.update_[i].OutputUnchecked(out)
        for i in range(len(self.insert_)):
            out.putVarInt32(26)
            out.putVarInt32(self.insert_[i].ByteSize())
            self.insert_[i].OutputUnchecked(out)
        for i in range(len(self.insert_auto_id_)):
            out.putVarInt32(34)
            out.putVarInt32(self.insert_auto_id_[i].ByteSize())
            self.insert_auto_id_[i].OutputUnchecked(out)
        for i in range(len(self.delete_)):
            out.putVarInt32(42)
            out.putVarInt32(self.delete_[i].ByteSize())
            self.delete_[i].OutputUnchecked(out)
        if self.has_force_:
            out.putVarInt32(48)
            out.putBoolean(self.force_)

    def OutputPartial(self, out):
        for i in range(len(self.upsert_)):
            out.putVarInt32(10)
            out.putVarInt32(self.upsert_[i].ByteSizePartial())
            self.upsert_[i].OutputPartial(out)
        for i in range(len(self.update_)):
            out.putVarInt32(18)
            out.putVarInt32(self.update_[i].ByteSizePartial())
            self.update_[i].OutputPartial(out)
        for i in range(len(self.insert_)):
            out.putVarInt32(26)
            out.putVarInt32(self.insert_[i].ByteSizePartial())
            self.insert_[i].OutputPartial(out)
        for i in range(len(self.insert_auto_id_)):
            out.putVarInt32(34)
            out.putVarInt32(self.insert_auto_id_[i].ByteSizePartial())
            self.insert_auto_id_[i].OutputPartial(out)
        for i in range(len(self.delete_)):
            out.putVarInt32(42)
            out.putVarInt32(self.delete_[i].ByteSizePartial())
            self.delete_[i].OutputPartial(out)
        if self.has_force_:
            out.putVarInt32(48)
            out.putBoolean(self.force_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_upsert().TryMerge(tmp)
                continue
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_update().TryMerge(tmp)
                continue
            if tt == 26:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_insert().TryMerge(tmp)
                continue
            if tt == 34:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_insert_auto_id().TryMerge(tmp)
                continue
            if tt == 42:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_delete().TryMerge(tmp)
                continue
            if tt == 48:
                self.set_force(d.getBoolean())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.upsert_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'upsert%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        cnt = 0
        for e in self.update_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'update%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        cnt = 0
        for e in self.insert_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'insert%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        cnt = 0
        for e in self.insert_auto_id_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'insert_auto_id%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        cnt = 0
        for e in self.delete_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'delete%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        if self.has_force_:
            res += prefix + 'force: %s\n' % self.DebugFormatBool(self.force_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kupsert = 1
    kupdate = 2
    kinsert = 3
    kinsert_auto_id = 4
    kdelete = 5
    kforce = 6
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'upsert', 2: 'update', 3: 'insert', 4: 'insert_auto_id', 5: 'delete', 6: 'force'}, 6)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.STRING, 4: ProtocolBuffer.Encoder.STRING, 5: ProtocolBuffer.Encoder.STRING, 6: ProtocolBuffer.Encoder.NUMERIC}, 6, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.datastore.v4.DeprecatedMutation'