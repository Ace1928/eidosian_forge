from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueQueryTasksResponse_Task(ProtocolBuffer.ProtocolMessage):
    GET = 1
    POST = 2
    HEAD = 3
    PUT = 4
    DELETE = 5
    _RequestMethod_NAMES = {1: 'GET', 2: 'POST', 3: 'HEAD', 4: 'PUT', 5: 'DELETE'}

    def RequestMethod_Name(cls, x):
        return cls._RequestMethod_NAMES.get(x, '')
    RequestMethod_Name = classmethod(RequestMethod_Name)
    has_task_name_ = 0
    task_name_ = ''
    has_eta_usec_ = 0
    eta_usec_ = 0
    has_url_ = 0
    url_ = ''
    has_method_ = 0
    method_ = 0
    has_retry_count_ = 0
    retry_count_ = 0
    has_body_size_ = 0
    body_size_ = 0
    has_body_ = 0
    body_ = ''
    has_creation_time_usec_ = 0
    creation_time_usec_ = 0
    has_crontimetable_ = 0
    crontimetable_ = None
    has_runlog_ = 0
    runlog_ = None
    has_description_ = 0
    description_ = ''
    has_payload_ = 0
    payload_ = None
    has_retry_parameters_ = 0
    retry_parameters_ = None
    has_first_try_usec_ = 0
    first_try_usec_ = 0
    has_tag_ = 0
    tag_ = ''
    has_execution_count_ = 0
    execution_count_ = 0

    def __init__(self, contents=None):
        self.header_ = []
        self.lazy_init_lock_ = _Lock()
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

    def url(self):
        return self.url_

    def set_url(self, x):
        self.has_url_ = 1
        self.url_ = x

    def clear_url(self):
        if self.has_url_:
            self.has_url_ = 0
            self.url_ = ''

    def has_url(self):
        return self.has_url_

    def method(self):
        return self.method_

    def set_method(self, x):
        self.has_method_ = 1
        self.method_ = x

    def clear_method(self):
        if self.has_method_:
            self.has_method_ = 0
            self.method_ = 0

    def has_method(self):
        return self.has_method_

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

    def header_size(self):
        return len(self.header_)

    def header_list(self):
        return self.header_

    def header(self, i):
        return self.header_[i]

    def mutable_header(self, i):
        return self.header_[i]

    def add_header(self):
        x = TaskQueueQueryTasksResponse_TaskHeader()
        self.header_.append(x)
        return x

    def clear_header(self):
        self.header_ = []

    def body_size(self):
        return self.body_size_

    def set_body_size(self, x):
        self.has_body_size_ = 1
        self.body_size_ = x

    def clear_body_size(self):
        if self.has_body_size_:
            self.has_body_size_ = 0
            self.body_size_ = 0

    def has_body_size(self):
        return self.has_body_size_

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

    def creation_time_usec(self):
        return self.creation_time_usec_

    def set_creation_time_usec(self, x):
        self.has_creation_time_usec_ = 1
        self.creation_time_usec_ = x

    def clear_creation_time_usec(self):
        if self.has_creation_time_usec_:
            self.has_creation_time_usec_ = 0
            self.creation_time_usec_ = 0

    def has_creation_time_usec(self):
        return self.has_creation_time_usec_

    def crontimetable(self):
        if self.crontimetable_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.crontimetable_ is None:
                    self.crontimetable_ = TaskQueueQueryTasksResponse_TaskCronTimetable()
            finally:
                self.lazy_init_lock_.release()
        return self.crontimetable_

    def mutable_crontimetable(self):
        self.has_crontimetable_ = 1
        return self.crontimetable()

    def clear_crontimetable(self):
        if self.has_crontimetable_:
            self.has_crontimetable_ = 0
            if self.crontimetable_ is not None:
                self.crontimetable_.Clear()

    def has_crontimetable(self):
        return self.has_crontimetable_

    def runlog(self):
        if self.runlog_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.runlog_ is None:
                    self.runlog_ = TaskQueueQueryTasksResponse_TaskRunLog()
            finally:
                self.lazy_init_lock_.release()
        return self.runlog_

    def mutable_runlog(self):
        self.has_runlog_ = 1
        return self.runlog()

    def clear_runlog(self):
        if self.has_runlog_:
            self.has_runlog_ = 0
            if self.runlog_ is not None:
                self.runlog_.Clear()

    def has_runlog(self):
        return self.has_runlog_

    def description(self):
        return self.description_

    def set_description(self, x):
        self.has_description_ = 1
        self.description_ = x

    def clear_description(self):
        if self.has_description_:
            self.has_description_ = 0
            self.description_ = ''

    def has_description(self):
        return self.has_description_

    def payload(self):
        if self.payload_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.payload_ is None:
                    self.payload_ = MessageSet()
            finally:
                self.lazy_init_lock_.release()
        return self.payload_

    def mutable_payload(self):
        self.has_payload_ = 1
        return self.payload()

    def clear_payload(self):
        if self.has_payload_:
            self.has_payload_ = 0
            if self.payload_ is not None:
                self.payload_.Clear()

    def has_payload(self):
        return self.has_payload_

    def retry_parameters(self):
        if self.retry_parameters_ is None:
            self.lazy_init_lock_.acquire()
            try:
                if self.retry_parameters_ is None:
                    self.retry_parameters_ = TaskQueueRetryParameters()
            finally:
                self.lazy_init_lock_.release()
        return self.retry_parameters_

    def mutable_retry_parameters(self):
        self.has_retry_parameters_ = 1
        return self.retry_parameters()

    def clear_retry_parameters(self):
        if self.has_retry_parameters_:
            self.has_retry_parameters_ = 0
            if self.retry_parameters_ is not None:
                self.retry_parameters_.Clear()

    def has_retry_parameters(self):
        return self.has_retry_parameters_

    def first_try_usec(self):
        return self.first_try_usec_

    def set_first_try_usec(self, x):
        self.has_first_try_usec_ = 1
        self.first_try_usec_ = x

    def clear_first_try_usec(self):
        if self.has_first_try_usec_:
            self.has_first_try_usec_ = 0
            self.first_try_usec_ = 0

    def has_first_try_usec(self):
        return self.has_first_try_usec_

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

    def execution_count(self):
        return self.execution_count_

    def set_execution_count(self, x):
        self.has_execution_count_ = 1
        self.execution_count_ = x

    def clear_execution_count(self):
        if self.has_execution_count_:
            self.has_execution_count_ = 0
            self.execution_count_ = 0

    def has_execution_count(self):
        return self.has_execution_count_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_task_name():
            self.set_task_name(x.task_name())
        if x.has_eta_usec():
            self.set_eta_usec(x.eta_usec())
        if x.has_url():
            self.set_url(x.url())
        if x.has_method():
            self.set_method(x.method())
        if x.has_retry_count():
            self.set_retry_count(x.retry_count())
        for i in range(x.header_size()):
            self.add_header().CopyFrom(x.header(i))
        if x.has_body_size():
            self.set_body_size(x.body_size())
        if x.has_body():
            self.set_body(x.body())
        if x.has_creation_time_usec():
            self.set_creation_time_usec(x.creation_time_usec())
        if x.has_crontimetable():
            self.mutable_crontimetable().MergeFrom(x.crontimetable())
        if x.has_runlog():
            self.mutable_runlog().MergeFrom(x.runlog())
        if x.has_description():
            self.set_description(x.description())
        if x.has_payload():
            self.mutable_payload().MergeFrom(x.payload())
        if x.has_retry_parameters():
            self.mutable_retry_parameters().MergeFrom(x.retry_parameters())
        if x.has_first_try_usec():
            self.set_first_try_usec(x.first_try_usec())
        if x.has_tag():
            self.set_tag(x.tag())
        if x.has_execution_count():
            self.set_execution_count(x.execution_count())

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
        if self.has_url_ != x.has_url_:
            return 0
        if self.has_url_ and self.url_ != x.url_:
            return 0
        if self.has_method_ != x.has_method_:
            return 0
        if self.has_method_ and self.method_ != x.method_:
            return 0
        if self.has_retry_count_ != x.has_retry_count_:
            return 0
        if self.has_retry_count_ and self.retry_count_ != x.retry_count_:
            return 0
        if len(self.header_) != len(x.header_):
            return 0
        for e1, e2 in zip(self.header_, x.header_):
            if e1 != e2:
                return 0
        if self.has_body_size_ != x.has_body_size_:
            return 0
        if self.has_body_size_ and self.body_size_ != x.body_size_:
            return 0
        if self.has_body_ != x.has_body_:
            return 0
        if self.has_body_ and self.body_ != x.body_:
            return 0
        if self.has_creation_time_usec_ != x.has_creation_time_usec_:
            return 0
        if self.has_creation_time_usec_ and self.creation_time_usec_ != x.creation_time_usec_:
            return 0
        if self.has_crontimetable_ != x.has_crontimetable_:
            return 0
        if self.has_crontimetable_ and self.crontimetable_ != x.crontimetable_:
            return 0
        if self.has_runlog_ != x.has_runlog_:
            return 0
        if self.has_runlog_ and self.runlog_ != x.runlog_:
            return 0
        if self.has_description_ != x.has_description_:
            return 0
        if self.has_description_ and self.description_ != x.description_:
            return 0
        if self.has_payload_ != x.has_payload_:
            return 0
        if self.has_payload_ and self.payload_ != x.payload_:
            return 0
        if self.has_retry_parameters_ != x.has_retry_parameters_:
            return 0
        if self.has_retry_parameters_ and self.retry_parameters_ != x.retry_parameters_:
            return 0
        if self.has_first_try_usec_ != x.has_first_try_usec_:
            return 0
        if self.has_first_try_usec_ and self.first_try_usec_ != x.first_try_usec_:
            return 0
        if self.has_tag_ != x.has_tag_:
            return 0
        if self.has_tag_ and self.tag_ != x.tag_:
            return 0
        if self.has_execution_count_ != x.has_execution_count_:
            return 0
        if self.has_execution_count_ and self.execution_count_ != x.execution_count_:
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
        for p in self.header_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        if not self.has_creation_time_usec_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: creation_time_usec not set.')
        if self.has_crontimetable_ and (not self.crontimetable_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_runlog_ and (not self.runlog_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_payload_ and (not self.payload_.IsInitialized(debug_strs)):
            initialized = 0
        if self.has_retry_parameters_ and (not self.retry_parameters_.IsInitialized(debug_strs)):
            initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.task_name_))
        n += self.lengthVarInt64(self.eta_usec_)
        if self.has_url_:
            n += 1 + self.lengthString(len(self.url_))
        if self.has_method_:
            n += 1 + self.lengthVarInt64(self.method_)
        if self.has_retry_count_:
            n += 1 + self.lengthVarInt64(self.retry_count_)
        n += 2 * len(self.header_)
        for i in range(len(self.header_)):
            n += self.header_[i].ByteSize()
        if self.has_body_size_:
            n += 1 + self.lengthVarInt64(self.body_size_)
        if self.has_body_:
            n += 1 + self.lengthString(len(self.body_))
        n += self.lengthVarInt64(self.creation_time_usec_)
        if self.has_crontimetable_:
            n += 2 + self.crontimetable_.ByteSize()
        if self.has_runlog_:
            n += 4 + self.runlog_.ByteSize()
        if self.has_description_:
            n += 2 + self.lengthString(len(self.description_))
        if self.has_payload_:
            n += 2 + self.lengthString(self.payload_.ByteSize())
        if self.has_retry_parameters_:
            n += 2 + self.lengthString(self.retry_parameters_.ByteSize())
        if self.has_first_try_usec_:
            n += 2 + self.lengthVarInt64(self.first_try_usec_)
        if self.has_tag_:
            n += 2 + self.lengthString(len(self.tag_))
        if self.has_execution_count_:
            n += 2 + self.lengthVarInt64(self.execution_count_)
        return n + 3

    def ByteSizePartial(self):
        n = 0
        if self.has_task_name_:
            n += 1
            n += self.lengthString(len(self.task_name_))
        if self.has_eta_usec_:
            n += 1
            n += self.lengthVarInt64(self.eta_usec_)
        if self.has_url_:
            n += 1 + self.lengthString(len(self.url_))
        if self.has_method_:
            n += 1 + self.lengthVarInt64(self.method_)
        if self.has_retry_count_:
            n += 1 + self.lengthVarInt64(self.retry_count_)
        n += 2 * len(self.header_)
        for i in range(len(self.header_)):
            n += self.header_[i].ByteSizePartial()
        if self.has_body_size_:
            n += 1 + self.lengthVarInt64(self.body_size_)
        if self.has_body_:
            n += 1 + self.lengthString(len(self.body_))
        if self.has_creation_time_usec_:
            n += 1
            n += self.lengthVarInt64(self.creation_time_usec_)
        if self.has_crontimetable_:
            n += 2 + self.crontimetable_.ByteSizePartial()
        if self.has_runlog_:
            n += 4 + self.runlog_.ByteSizePartial()
        if self.has_description_:
            n += 2 + self.lengthString(len(self.description_))
        if self.has_payload_:
            n += 2 + self.lengthString(self.payload_.ByteSizePartial())
        if self.has_retry_parameters_:
            n += 2 + self.lengthString(self.retry_parameters_.ByteSizePartial())
        if self.has_first_try_usec_:
            n += 2 + self.lengthVarInt64(self.first_try_usec_)
        if self.has_tag_:
            n += 2 + self.lengthString(len(self.tag_))
        if self.has_execution_count_:
            n += 2 + self.lengthVarInt64(self.execution_count_)
        return n

    def Clear(self):
        self.clear_task_name()
        self.clear_eta_usec()
        self.clear_url()
        self.clear_method()
        self.clear_retry_count()
        self.clear_header()
        self.clear_body_size()
        self.clear_body()
        self.clear_creation_time_usec()
        self.clear_crontimetable()
        self.clear_runlog()
        self.clear_description()
        self.clear_payload()
        self.clear_retry_parameters()
        self.clear_first_try_usec()
        self.clear_tag()
        self.clear_execution_count()

    def OutputUnchecked(self, out):
        out.putVarInt32(18)
        out.putPrefixedString(self.task_name_)
        out.putVarInt32(24)
        out.putVarInt64(self.eta_usec_)
        if self.has_url_:
            out.putVarInt32(34)
            out.putPrefixedString(self.url_)
        if self.has_method_:
            out.putVarInt32(40)
            out.putVarInt32(self.method_)
        if self.has_retry_count_:
            out.putVarInt32(48)
            out.putVarInt32(self.retry_count_)
        for i in range(len(self.header_)):
            out.putVarInt32(59)
            self.header_[i].OutputUnchecked(out)
            out.putVarInt32(60)
        if self.has_body_size_:
            out.putVarInt32(80)
            out.putVarInt32(self.body_size_)
        if self.has_body_:
            out.putVarInt32(90)
            out.putPrefixedString(self.body_)
        out.putVarInt32(96)
        out.putVarInt64(self.creation_time_usec_)
        if self.has_crontimetable_:
            out.putVarInt32(107)
            self.crontimetable_.OutputUnchecked(out)
            out.putVarInt32(108)
        if self.has_runlog_:
            out.putVarInt32(131)
            self.runlog_.OutputUnchecked(out)
            out.putVarInt32(132)
        if self.has_description_:
            out.putVarInt32(170)
            out.putPrefixedString(self.description_)
        if self.has_payload_:
            out.putVarInt32(178)
            out.putVarInt32(self.payload_.ByteSize())
            self.payload_.OutputUnchecked(out)
        if self.has_retry_parameters_:
            out.putVarInt32(186)
            out.putVarInt32(self.retry_parameters_.ByteSize())
            self.retry_parameters_.OutputUnchecked(out)
        if self.has_first_try_usec_:
            out.putVarInt32(192)
            out.putVarInt64(self.first_try_usec_)
        if self.has_tag_:
            out.putVarInt32(202)
            out.putPrefixedString(self.tag_)
        if self.has_execution_count_:
            out.putVarInt32(208)
            out.putVarInt32(self.execution_count_)

    def OutputPartial(self, out):
        if self.has_task_name_:
            out.putVarInt32(18)
            out.putPrefixedString(self.task_name_)
        if self.has_eta_usec_:
            out.putVarInt32(24)
            out.putVarInt64(self.eta_usec_)
        if self.has_url_:
            out.putVarInt32(34)
            out.putPrefixedString(self.url_)
        if self.has_method_:
            out.putVarInt32(40)
            out.putVarInt32(self.method_)
        if self.has_retry_count_:
            out.putVarInt32(48)
            out.putVarInt32(self.retry_count_)
        for i in range(len(self.header_)):
            out.putVarInt32(59)
            self.header_[i].OutputPartial(out)
            out.putVarInt32(60)
        if self.has_body_size_:
            out.putVarInt32(80)
            out.putVarInt32(self.body_size_)
        if self.has_body_:
            out.putVarInt32(90)
            out.putPrefixedString(self.body_)
        if self.has_creation_time_usec_:
            out.putVarInt32(96)
            out.putVarInt64(self.creation_time_usec_)
        if self.has_crontimetable_:
            out.putVarInt32(107)
            self.crontimetable_.OutputPartial(out)
            out.putVarInt32(108)
        if self.has_runlog_:
            out.putVarInt32(131)
            self.runlog_.OutputPartial(out)
            out.putVarInt32(132)
        if self.has_description_:
            out.putVarInt32(170)
            out.putPrefixedString(self.description_)
        if self.has_payload_:
            out.putVarInt32(178)
            out.putVarInt32(self.payload_.ByteSizePartial())
            self.payload_.OutputPartial(out)
        if self.has_retry_parameters_:
            out.putVarInt32(186)
            out.putVarInt32(self.retry_parameters_.ByteSizePartial())
            self.retry_parameters_.OutputPartial(out)
        if self.has_first_try_usec_:
            out.putVarInt32(192)
            out.putVarInt64(self.first_try_usec_)
        if self.has_tag_:
            out.putVarInt32(202)
            out.putPrefixedString(self.tag_)
        if self.has_execution_count_:
            out.putVarInt32(208)
            out.putVarInt32(self.execution_count_)

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
            if tt == 34:
                self.set_url(d.getPrefixedString())
                continue
            if tt == 40:
                self.set_method(d.getVarInt32())
                continue
            if tt == 48:
                self.set_retry_count(d.getVarInt32())
                continue
            if tt == 59:
                self.add_header().TryMerge(d)
                continue
            if tt == 80:
                self.set_body_size(d.getVarInt32())
                continue
            if tt == 90:
                self.set_body(d.getPrefixedString())
                continue
            if tt == 96:
                self.set_creation_time_usec(d.getVarInt64())
                continue
            if tt == 107:
                self.mutable_crontimetable().TryMerge(d)
                continue
            if tt == 131:
                self.mutable_runlog().TryMerge(d)
                continue
            if tt == 170:
                self.set_description(d.getPrefixedString())
                continue
            if tt == 178:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_payload().TryMerge(tmp)
                continue
            if tt == 186:
                length = d.getVarInt32()
                tmp = ProtocolBuffer.Decoder(d.buffer(), d.pos(), d.pos() + length)
                d.skip(length)
                self.mutable_retry_parameters().TryMerge(tmp)
                continue
            if tt == 192:
                self.set_first_try_usec(d.getVarInt64())
                continue
            if tt == 202:
                self.set_tag(d.getPrefixedString())
                continue
            if tt == 208:
                self.set_execution_count(d.getVarInt32())
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
        if self.has_url_:
            res += prefix + 'url: %s\n' % self.DebugFormatString(self.url_)
        if self.has_method_:
            res += prefix + 'method: %s\n' % self.DebugFormatInt32(self.method_)
        if self.has_retry_count_:
            res += prefix + 'retry_count: %s\n' % self.DebugFormatInt32(self.retry_count_)
        cnt = 0
        for e in self.header_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'Header%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        if self.has_body_size_:
            res += prefix + 'body_size: %s\n' % self.DebugFormatInt32(self.body_size_)
        if self.has_body_:
            res += prefix + 'body: %s\n' % self.DebugFormatString(self.body_)
        if self.has_creation_time_usec_:
            res += prefix + 'creation_time_usec: %s\n' % self.DebugFormatInt64(self.creation_time_usec_)
        if self.has_crontimetable_:
            res += prefix + 'CronTimetable {\n'
            res += self.crontimetable_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
        if self.has_runlog_:
            res += prefix + 'RunLog {\n'
            res += self.runlog_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
        if self.has_description_:
            res += prefix + 'description: %s\n' % self.DebugFormatString(self.description_)
        if self.has_payload_:
            res += prefix + 'payload <\n'
            res += self.payload_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_retry_parameters_:
            res += prefix + 'retry_parameters <\n'
            res += self.retry_parameters_.__str__(prefix + '  ', printElemNumber)
            res += prefix + '>\n'
        if self.has_first_try_usec_:
            res += prefix + 'first_try_usec: %s\n' % self.DebugFormatInt64(self.first_try_usec_)
        if self.has_tag_:
            res += prefix + 'tag: %s\n' % self.DebugFormatString(self.tag_)
        if self.has_execution_count_:
            res += prefix + 'execution_count: %s\n' % self.DebugFormatInt32(self.execution_count_)
        return res