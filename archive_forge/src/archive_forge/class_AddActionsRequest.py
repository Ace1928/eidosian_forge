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
class AddActionsRequest(ProtocolBuffer.ProtocolMessage):
    has_transaction_ = 0

    def __init__(self, contents=None):
        self.transaction_ = Transaction()
        self.action_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def transaction(self):
        return self.transaction_

    def mutable_transaction(self):
        self.has_transaction_ = 1
        return self.transaction_

    def clear_transaction(self):
        self.has_transaction_ = 0
        self.transaction_.Clear()

    def has_transaction(self):
        return self.has_transaction_

    def action_size(self):
        return len(self.action_)

    def action_list(self):
        return self.action_

    def action(self, i):
        return self.action_[i]

    def mutable_action(self, i):
        return self.action_[i]

    def add_action(self):
        x = googlecloudsdk.third_party.appengine.googlestorage.onestore.v3.action_pb.Action()
        self.action_.append(x)
        return x

    def clear_action(self):
        self.action_ = []

    def MergeFrom(self, x):
        assert x is not self
        if x.has_transaction():
            self.mutable_transaction().MergeFrom(x.transaction())
        for i in range(x.action_size()):
            self.add_action().CopyFrom(x.action(i))

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_transaction_ != x.has_transaction_:
            return 0
        if self.has_transaction_ and self.transaction_ != x.transaction_:
            return 0
        if len(self.action_) != len(x.action_):
            return 0
        for e1, e2 in zip(self.action_, x.action_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_transaction_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: transaction not set.')
        elif not self.transaction_.IsInitialized(debug_strs):
            initialized = 0
        for p in self.action_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(self.transaction_.ByteSize())
        n += 1 * len(self.action_)
        for i in range(len(self.action_)):
            n += self.lengthString(self.action_[i].ByteSize())
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_transaction_:
            n += 1
            n += self.lengthString(self.transaction_.ByteSizePartial())
        n += 1 * len(self.action_)
        for i in range(len(self.action_)):
            n += self.lengthString(self.action_[i].ByteSizePartial())
        return n

    def Clear(self):
        self.clear_transaction()
        self.clear_action()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putVarInt32(self.transaction_.ByteSize())
        self.transaction_.OutputUnchecked(out)
        for i in range(len(self.action_)):
            out.putVarInt32(18)
            out.putVarInt32(self.action_[i].ByteSize())
            self.action_[i].OutputUnchecked(out)

    def OutputPartial(self, out):
        if self.has_transaction_:
            out.putVarInt32(10)
            out.putVarInt32(self.transaction_.ByteSizePartial())
            self.transaction_.OutputPartial(out)
        for i in range(len(self.action_)):
            out.putVarInt32(18)
            out.putVarInt32(self.action_[i].ByteSizePartial())
            self.action_[i].OutputPartial(out)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_transaction().TryMerge(tmp)
                continue
            if tt == 18:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.add_action().TryMerge(tmp)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_transaction_:
            res += prefix + 'transaction <\n'
            res += self.transaction_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        cnt = 0
        for e in self.action_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'action%s <\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    ktransaction = 1
    kaction = 2
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'transaction', 2: 'action'}, 2)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING}, 2, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting_datastore_v3.AddActionsRequest'