from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueForceRunRequest(ProtocolBuffer.ProtocolMessage):
    has_app_id_ = 0
    app_id_ = ''
    has_queue_name_ = 0
    queue_name_ = ''
    has_task_name_ = 0
    task_name_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

    def app_id(self):
        return self.app_id_

    def set_app_id(self, x):
        self.has_app_id_ = 1
        self.app_id_ = x

    def clear_app_id(self):
        if self.has_app_id_:
            self.has_app_id_ = 0
            self.app_id_ = ''

    def has_app_id(self):
        return self.has_app_id_

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

    def MergeFrom(self, x):
        assert x is not self
        if x.has_app_id():
            self.set_app_id(x.app_id())
        if x.has_queue_name():
            self.set_queue_name(x.queue_name())
        if x.has_task_name():
            self.set_task_name(x.task_name())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_app_id_ != x.has_app_id_:
            return 0
        if self.has_app_id_ and self.app_id_ != x.app_id_:
            return 0
        if self.has_queue_name_ != x.has_queue_name_:
            return 0
        if self.has_queue_name_ and self.queue_name_ != x.queue_name_:
            return 0
        if self.has_task_name_ != x.has_task_name_:
            return 0
        if self.has_task_name_ and self.task_name_ != x.task_name_:
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
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        n += self.lengthString(len(self.queue_name_))
        n += self.lengthString(len(self.task_name_))
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        if self.has_queue_name_:
            n += 1
            n += self.lengthString(len(self.queue_name_))
        if self.has_task_name_:
            n += 1
            n += self.lengthString(len(self.task_name_))
        return n

    def Clear(self):
        self.clear_app_id()
        self.clear_queue_name()
        self.clear_task_name()

    def OutputUnchecked(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        out.putVarInt32(18)
        out.putPrefixedString(self.queue_name_)
        out.putVarInt32(26)
        out.putPrefixedString(self.task_name_)

    def OutputPartial(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        if self.has_queue_name_:
            out.putVarInt32(18)
            out.putPrefixedString(self.queue_name_)
        if self.has_task_name_:
            out.putVarInt32(26)
            out.putPrefixedString(self.task_name_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_app_id(d.getPrefixedString())
                continue
            if tt == 18:
                self.set_queue_name(d.getPrefixedString())
                continue
            if tt == 26:
                self.set_task_name(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_app_id_:
            res += prefix + 'app_id: %s\n' % self.DebugFormatString(self.app_id_)
        if self.has_queue_name_:
            res += prefix + 'queue_name: %s\n' % self.DebugFormatString(self.queue_name_)
        if self.has_task_name_:
            res += prefix + 'task_name: %s\n' % self.DebugFormatString(self.task_name_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kapp_id = 1
    kqueue_name = 2
    ktask_name = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'app_id', 2: 'queue_name', 3: 'task_name'}, 3)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.STRING}, 3, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueForceRunRequest'