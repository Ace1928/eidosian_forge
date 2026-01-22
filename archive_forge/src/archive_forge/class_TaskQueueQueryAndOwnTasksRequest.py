from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueQueryAndOwnTasksRequest(ProtocolBuffer.ProtocolMessage):
    has_queue_name_ = 0
    queue_name_ = ''
    has_lease_seconds_ = 0
    lease_seconds_ = 0.0
    has_max_tasks_ = 0
    max_tasks_ = 0
    has_group_by_tag_ = 0
    group_by_tag_ = 0
    has_tag_ = 0
    tag_ = ''

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

    def max_tasks(self):
        return self.max_tasks_

    def set_max_tasks(self, x):
        self.has_max_tasks_ = 1
        self.max_tasks_ = x

    def clear_max_tasks(self):
        if self.has_max_tasks_:
            self.has_max_tasks_ = 0
            self.max_tasks_ = 0

    def has_max_tasks(self):
        return self.has_max_tasks_

    def group_by_tag(self):
        return self.group_by_tag_

    def set_group_by_tag(self, x):
        self.has_group_by_tag_ = 1
        self.group_by_tag_ = x

    def clear_group_by_tag(self):
        if self.has_group_by_tag_:
            self.has_group_by_tag_ = 0
            self.group_by_tag_ = 0

    def has_group_by_tag(self):
        return self.has_group_by_tag_

    def tag(self):
        return self.tag_

    def set_tag(self, x):
        self.has_tag_ = 1
        self.tag_ = x

    def clear_tag(self):
        if self.has_tag_:
            self.has_tag_ = 0
            self.tag_ = ''

    def has_tag(self):
        return self.has_tag_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_queue_name():
            self.set_queue_name(x.queue_name())
        if x.has_lease_seconds():
            self.set_lease_seconds(x.lease_seconds())
        if x.has_max_tasks():
            self.set_max_tasks(x.max_tasks())
        if x.has_group_by_tag():
            self.set_group_by_tag(x.group_by_tag())
        if x.has_tag():
            self.set_tag(x.tag())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_queue_name_ != x.has_queue_name_:
            return 0
        if self.has_queue_name_ and self.queue_name_ != x.queue_name_:
            return 0
        if self.has_lease_seconds_ != x.has_lease_seconds_:
            return 0
        if self.has_lease_seconds_ and self.lease_seconds_ != x.lease_seconds_:
            return 0
        if self.has_max_tasks_ != x.has_max_tasks_:
            return 0
        if self.has_max_tasks_ and self.max_tasks_ != x.max_tasks_:
            return 0
        if self.has_group_by_tag_ != x.has_group_by_tag_:
            return 0
        if self.has_group_by_tag_ and self.group_by_tag_ != x.group_by_tag_:
            return 0
        if self.has_tag_ != x.has_tag_:
            return 0
        if self.has_tag_ and self.tag_ != x.tag_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_queue_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: queue_name not set.')
        if not self.has_lease_seconds_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: lease_seconds not set.')
        if not self.has_max_tasks_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: max_tasks not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.queue_name_))
        n += self.lengthVarInt64(self.max_tasks_)
        if self.has_group_by_tag_:
            n += 2
        if self.has_tag_:
            n += 1 + self.lengthString(len(self.tag_))
        return n + 11

    def ByteSizePartial(self):
        n = 0
        if self.has_queue_name_:
            n += 1
            n += self.lengthString(len(self.queue_name_))
        if self.has_lease_seconds_:
            n += 9
        if self.has_max_tasks_:
            n += 1
            n += self.lengthVarInt64(self.max_tasks_)
        if self.has_group_by_tag_:
            n += 2
        if self.has_tag_:
            n += 1 + self.lengthString(len(self.tag_))
        return n

    def Clear(self):
        self.clear_queue_name()
        self.clear_lease_seconds()
        self.clear_max_tasks()
        self.clear_group_by_tag()
        self.clear_tag()

    def OutputUnchecked(self, out):
        out.putVarInt32(10)
        out.putPrefixedString(self.queue_name_)
        out.putVarInt32(17)
        out.putDouble(self.lease_seconds_)
        out.putVarInt32(24)
        out.putVarInt64(self.max_tasks_)
        if self.has_group_by_tag_:
            out.putVarInt32(32)
            out.putBoolean(self.group_by_tag_)
        if self.has_tag_:
            out.putVarInt32(42)
            out.putPrefixedString(self.tag_)

    def OutputPartial(self, out):
        if self.has_queue_name_:
            out.putVarInt32(10)
            out.putPrefixedString(self.queue_name_)
        if self.has_lease_seconds_:
            out.putVarInt32(17)
            out.putDouble(self.lease_seconds_)
        if self.has_max_tasks_:
            out.putVarInt32(24)
            out.putVarInt64(self.max_tasks_)
        if self.has_group_by_tag_:
            out.putVarInt32(32)
            out.putBoolean(self.group_by_tag_)
        if self.has_tag_:
            out.putVarInt32(42)
            out.putPrefixedString(self.tag_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_queue_name(d.getPrefixedString())
                continue
            if tt == 17:
                self.set_lease_seconds(d.getDouble())
                continue
            if tt == 24:
                self.set_max_tasks(d.getVarInt64())
                continue
            if tt == 32:
                self.set_group_by_tag(d.getBoolean())
                continue
            if tt == 42:
                self.set_tag(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_queue_name_:
            res += prefix + 'queue_name: %s\n' % self.DebugFormatString(self.queue_name_)
        if self.has_lease_seconds_:
            res += prefix + 'lease_seconds: %s\n' % self.DebugFormat(self.lease_seconds_)
        if self.has_max_tasks_:
            res += prefix + 'max_tasks: %s\n' % self.DebugFormatInt64(self.max_tasks_)
        if self.has_group_by_tag_:
            res += prefix + 'group_by_tag: %s\n' % self.DebugFormatBool(self.group_by_tag_)
        if self.has_tag_:
            res += prefix + 'tag: %s\n' % self.DebugFormatString(self.tag_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kqueue_name = 1
    klease_seconds = 2
    kmax_tasks = 3
    kgroup_by_tag = 4
    ktag = 5
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'queue_name', 2: 'lease_seconds', 3: 'max_tasks', 4: 'group_by_tag', 5: 'tag'}, 5)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.DOUBLE, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.NUMERIC, 5: ProtocolBuffer.Encoder.STRING}, 5, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueQueryAndOwnTasksRequest'