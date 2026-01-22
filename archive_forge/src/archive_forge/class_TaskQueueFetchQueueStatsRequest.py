from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueFetchQueueStatsRequest(ProtocolBuffer.ProtocolMessage):
    has_app_id_ = 0
    app_id_ = ''
    has_max_num_tasks_ = 0
    max_num_tasks_ = 0

    def __init__(self, contents=None):
        self.queue_name_ = []
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

    def queue_name_size(self):
        return len(self.queue_name_)

    def queue_name_list(self):
        return self.queue_name_

    def queue_name(self, i):
        return self.queue_name_[i]

    def set_queue_name(self, i, x):
        self.queue_name_[i] = x

    def add_queue_name(self, x):
        self.queue_name_.append(x)

    def clear_queue_name(self):
        self.queue_name_ = []

    def max_num_tasks(self):
        return self.max_num_tasks_

    def set_max_num_tasks(self, x):
        self.has_max_num_tasks_ = 1
        self.max_num_tasks_ = x

    def clear_max_num_tasks(self):
        if self.has_max_num_tasks_:
            self.has_max_num_tasks_ = 0
            self.max_num_tasks_ = 0

    def has_max_num_tasks(self):
        return self.has_max_num_tasks_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_app_id():
            self.set_app_id(x.app_id())
        for i in range(x.queue_name_size()):
            self.add_queue_name(x.queue_name(i))
        if x.has_max_num_tasks():
            self.set_max_num_tasks(x.max_num_tasks())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_app_id_ != x.has_app_id_:
            return 0
        if self.has_app_id_ and self.app_id_ != x.app_id_:
            return 0
        if len(self.queue_name_) != len(x.queue_name_):
            return 0
        for e1, e2 in zip(self.queue_name_, x.queue_name_):
            if e1 != e2:
                return 0
        if self.has_max_num_tasks_ != x.has_max_num_tasks_:
            return 0
        if self.has_max_num_tasks_ and self.max_num_tasks_ != x.max_num_tasks_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        n += 1 * len(self.queue_name_)
        for i in range(len(self.queue_name_)):
            n += self.lengthString(len(self.queue_name_[i]))
        if self.has_max_num_tasks_:
            n += 1 + self.lengthVarInt64(self.max_num_tasks_)
        return n

    def ByteSizePartial(self):
        n = 0
        if self.has_app_id_:
            n += 1 + self.lengthString(len(self.app_id_))
        n += 1 * len(self.queue_name_)
        for i in range(len(self.queue_name_)):
            n += self.lengthString(len(self.queue_name_[i]))
        if self.has_max_num_tasks_:
            n += 1 + self.lengthVarInt64(self.max_num_tasks_)
        return n

    def Clear(self):
        self.clear_app_id()
        self.clear_queue_name()
        self.clear_max_num_tasks()

    def OutputUnchecked(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        for i in range(len(self.queue_name_)):
            out.putVarInt32(18)
            out.putPrefixedString(self.queue_name_[i])
        if self.has_max_num_tasks_:
            out.putVarInt32(24)
            out.putVarInt32(self.max_num_tasks_)

    def OutputPartial(self, out):
        if self.has_app_id_:
            out.putVarInt32(10)
            out.putPrefixedString(self.app_id_)
        for i in range(len(self.queue_name_)):
            out.putVarInt32(18)
            out.putPrefixedString(self.queue_name_[i])
        if self.has_max_num_tasks_:
            out.putVarInt32(24)
            out.putVarInt32(self.max_num_tasks_)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 10:
                self.set_app_id(d.getPrefixedString())
                continue
            if tt == 18:
                self.add_queue_name(d.getPrefixedString())
                continue
            if tt == 24:
                self.set_max_num_tasks(d.getVarInt32())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_app_id_:
            res += prefix + 'app_id: %s\n' % self.DebugFormatString(self.app_id_)
        cnt = 0
        for e in self.queue_name_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'queue_name%s: %s\n' % (elm, self.DebugFormatString(e))
            cnt += 1
        if self.has_max_num_tasks_:
            res += prefix + 'max_num_tasks: %s\n' % self.DebugFormatInt32(self.max_num_tasks_)
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kapp_id = 1
    kqueue_name = 2
    kmax_num_tasks = 3
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'app_id', 2: 'queue_name', 3: 'max_num_tasks'}, 3)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STRING, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC}, 3, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueFetchQueueStatsRequest'