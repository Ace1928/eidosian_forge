from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueQueryAndOwnTasksResponse_Task(ProtocolBuffer.ProtocolMessage):
    has_task_name_ = 0
    task_name_ = ''
    has_eta_usec_ = 0
    eta_usec_ = 0
    has_retry_count_ = 0
    retry_count_ = 0
    has_body_ = 0
    body_ = ''
    has_tag_ = 0
    tag_ = ''

    def __init__(self, contents=None):
        if contents is not None:
            self.MergeFromString(contents)

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

    def retry_count(self):
        return self.retry_count_

    def set_retry_count(self, x):
        self.has_retry_count_ = 1
        self.retry_count_ = x

    def clear_retry_count(self):
        if self.has_retry_count_:
            self.has_retry_count_ = 0
            self.retry_count_ = 0

    def has_retry_count(self):
        return self.has_retry_count_

    def body(self):
        return self.body_

    def set_body(self, x):
        self.has_body_ = 1
        self.body_ = x

    def clear_body(self):
        if self.has_body_:
            self.has_body_ = 0
            self.body_ = ''

    def has_body(self):
        return self.has_body_

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
        if x.has_task_name():
            self.set_task_name(x.task_name())
        if x.has_eta_usec():
            self.set_eta_usec(x.eta_usec())
        if x.has_retry_count():
            self.set_retry_count(x.retry_count())
        if x.has_body():
            self.set_body(x.body())
        if x.has_tag():
            self.set_tag(x.tag())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_task_name_ != x.has_task_name_:
            return 0
        if self.has_task_name_ and self.task_name_ != x.task_name_:
            return 0
        if self.has_eta_usec_ != x.has_eta_usec_:
            return 0
        if self.has_eta_usec_ and self.eta_usec_ != x.eta_usec_:
            return 0
        if self.has_retry_count_ != x.has_retry_count_:
            return 0
        if self.has_retry_count_ and self.retry_count_ != x.retry_count_:
            return 0
        if self.has_body_ != x.has_body_:
            return 0
        if self.has_body_ and self.body_ != x.body_:
            return 0
        if self.has_tag_ != x.has_tag_:
            return 0
        if self.has_tag_ and self.tag_ != x.tag_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_task_name_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: task_name not set.')
        if not self.has_eta_usec_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: eta_usec not set.')
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.task_name_))
        n += self.lengthVarInt64(self.eta_usec_)
        if self.has_retry_count_:
            n += 1 + self.lengthVarInt64(self.retry_count_)
        if self.has_body_:
            n += 1 + self.lengthString(len(self.body_))
        if self.has_tag_:
            n += 1 + self.lengthString(len(self.tag_))
        return n + 2

    def ByteSizePartial(self):
        n = 0
        if self.has_task_name_:
            n += 1
            n += self.lengthString(len(self.task_name_))
        if self.has_eta_usec_:
            n += 1
            n += self.lengthVarInt64(self.eta_usec_)
        if self.has_retry_count_:
            n += 1 + self.lengthVarInt64(self.retry_count_)
        if self.has_body_:
            n += 1 + self.lengthString(len(self.body_))
        if self.has_tag_:
            n += 1 + self.lengthString(len(self.tag_))
        return n

    def Clear(self):
        self.clear_task_name()
        self.clear_eta_usec()
        self.clear_retry_count()
        self.clear_body()
        self.clear_tag()

    def OutputUnchecked(self, out):
        out.putVarInt32(18)
        out.putPrefixedString(self.task_name_)
        out.putVarInt32(24)
        out.putVarInt64(self.eta_usec_)
        if self.has_retry_count_:
            out.putVarInt32(32)
            out.putVarInt32(self.retry_count_)
        if self.has_body_:
            out.putVarInt32(42)
            out.putPrefixedString(self.body_)
        if self.has_tag_:
            out.putVarInt32(50)
            out.putPrefixedString(self.tag_)

    def OutputPartial(self, out):
        if self.has_task_name_:
            out.putVarInt32(18)
            out.putPrefixedString(self.task_name_)
        if self.has_eta_usec_:
            out.putVarInt32(24)
            out.putVarInt64(self.eta_usec_)
        if self.has_retry_count_:
            out.putVarInt32(32)
            out.putVarInt32(self.retry_count_)
        if self.has_body_:
            out.putVarInt32(42)
            out.putPrefixedString(self.body_)
        if self.has_tag_:
            out.putVarInt32(50)
            out.putPrefixedString(self.tag_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 12:
                break
            if tt == 18:
                self.set_task_name(d.getPrefixedString())
                continue
            if tt == 24:
                self.set_eta_usec(d.getVarInt64())
                continue
            if tt == 32:
                self.set_retry_count(d.getVarInt32())
                continue
            if tt == 42:
                self.set_body(d.getPrefixedString())
                continue
            if tt == 50:
                self.set_tag(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_task_name_:
            res += prefix + 'task_name: %s\n' % self.DebugFormatString(self.task_name_)
        if self.has_eta_usec_:
            res += prefix + 'eta_usec: %s\n' % self.DebugFormatInt64(self.eta_usec_)
        if self.has_retry_count_:
            res += prefix + 'retry_count: %s\n' % self.DebugFormatInt32(self.retry_count_)
        if self.has_body_:
            res += prefix + 'body: %s\n' % self.DebugFormatString(self.body_)
        if self.has_tag_:
            res += prefix + 'tag: %s\n' % self.DebugFormatString(self.tag_)
        return res