from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueModifyTaskLeaseRequest(ProtocolBuffer.ProtocolMessage):
    has_queue_name_ = 0
    queue_name_ = ''
    has_task_name_ = 0
    task_name_ = ''
    has_eta_usec_ = 0
    eta_usec_ = 0
    has_lease_seconds_ = 0
    lease_seconds_ = 0.0

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def queue_name(self):
        return self.queue_name_

    def set_queue_name(self, x):
        self.has_queue_name_ = 1
        self.queue_name_ = x

    def clear_queue_name(self):
        if self.has_queue_name_:
            self.has_queue_name_ = 0
            self.queue_name_ = ''

    def has_queue_name(self):
        return self.has_queue_name_

    def task_name(self):
        return self.task_name_

    def set_task_name(self, x):
        self.has_task_name_ = 1
        self.task_name_ = x

    def clear_task_name(self):
        if self.has_task_name_:
            self.has_task_name_ = 0
            self.task_name_ = ''

    def has_task_name(self):
        return self.has_task_name_

    def eta_usec(self):
        return self.eta_usec_

    def set_eta_usec(self, x):
        self.has_eta_usec_ = 1
        self.eta_usec_ = x

    def clear_eta_usec(self):
        if self.has_eta_usec_:
            self.has_eta_usec_ = 0
            self.eta_usec_ = 0

    def has_eta_usec(self):
        return self.has_eta_usec_

    def lease_seconds(self):
        return self.lease_seconds_

    def set_lease_seconds(self, x):
        self.has_lease_seconds_ = 1
        self.lease_seconds_ = x

    def clear_lease_seconds(self):
        if self.has_lease_seconds_:
            self.has_lease_seconds_ = 0
            self.lease_seconds_ = 0.0

    def has_lease_seconds(self):
        return self.has_lease_seconds_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_queue_name():
            self.set_queue_name(x.queue_name())
        if x.has_task_name():
            self.set_task_name(x.task_name())
        if x.has_eta_usec():
            self.set_eta_usec(x.eta_usec())
        if x.has_lease_seconds():
            self.set_lease_seconds(x.lease_seconds())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_queue_name_ != x.has_queue_name_:
            return 0
        if self.has_queue_name_ and self.queue_name_ != x.queue_name_:
            return 0
        if self.has_task_name_ != x.has_task_name_:
            return 0
        if self.has_task_name_ and self.task_name_ != x.task_name_:
            return 0
        if self.has_eta_usec_ != x.has_eta_usec_:
            return 0
        if self.has_eta_usec_ and self.eta_usec_ != x.eta_usec_:
            return 0
        if self.has_lease_seconds_ != x.has_lease_seconds_:
            return 0
        if self.has_lease_seconds_ and self.lease_seconds_ != x.lease_seconds_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_queue_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: queue_name not set.')
        if not self.has_task_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: task_name not set.')
        if not self.has_eta_usec_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: eta_usec not set.')
        if not self.has_lease_seconds_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: lease_seconds not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.queue_name_))
        n += self.lengthString(len(self.task_name_))
        n += self.lengthVarInt64(self.eta_usec_)
        return n + 12

    def ByteSizePartial(self):
        n = 0
        if self.has_queue_name_:
            n += 1
            n += self.lengthString(len(self.queue_name_))
        if self.has_task_name_:
            n += 1
            n += self.lengthString(len(self.task_name_))
        if self.has_eta_usec_:
            n += 1
            n += self.lengthVarInt64(self.eta_usec_)
        if self.has_lease_seconds_:
            n += 9
        return n

    def Clear(self):
        self.clear_queue_name()
        self.clear_task_name()
        self.clear_eta_usec()
        self.clear_lease_seconds()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.queue_name_)
        out.putVarInt32(18)
        out.putPrefixedString(self.task_name_)
        out.putVarInt32(24)
        out.putVarInt64(self.eta_usec_)
        out.putVarInt32(33)
        out.putDouble(self.lease_seconds_)

    def OutputPartial(self, out):
        if self.has_queue_name_:
            out.putVarInt32(10)
            out.putPrefixedString(self.queue_name_)
        if self.has_task_name_:
            out.putVarInt32(18)
            out.putPrefixedString(self.task_name_)
        if self.has_eta_usec_:
            out.putVarInt32(24)
            out.putVarInt64(self.eta_usec_)
        if self.has_lease_seconds_:
            out.putVarInt32(33)
            out.putDouble(self.lease_seconds_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_queue_name(d.getPrefixedString())
                continue
            if tt == 18:
                self.set_task_name(d.getPrefixedString())
                continue
            if tt == 24:
                self.set_eta_usec(d.getVarInt64())
                continue
            if tt == 33:
                self.set_lease_seconds(d.getDouble())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_queue_name_:
            res += prefix + 'queue_name: %s\n' % self.DebugFormatString(self.queue_name_)
        if self.has_task_name_:
            res += prefix + 'task_name: %s\n' % self.DebugFormatString(self.task_name_)
        if self.has_eta_usec_:
            res += prefix + 'eta_usec: %s\n' % self.DebugFormatInt64(self.eta_usec_)
        if self.has_lease_seconds_:
            res += prefix + 'lease_seconds: %s\n' % self.DebugFormat(self.lease_seconds_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kqueue_name = 1
    ktask_name = 2
    keta_usec = 3
    klease_seconds = 4
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'queue_name', 2: 'task_name', 3: 'eta_usec', 4: 'lease_seconds'}, 4)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.DOUBLE}, 4, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueModifyTaskLeaseRequest'