from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueQueryTasksResponse(ProtocolBuffer.ProtocolMessage):

    def __init__(self, contents=None):
        self.task_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def task_size(self):
        return len(self.task_)

    def task_list(self):
        return self.task_

    def task(self, i):
        return self.task_[i]

    def mutable_task(self, i):
        return self.task_[i]

    def add_task(self):
        x = TaskQueueQueryTasksResponse_Task()
        self.task_.append(x)
        return x

    def clear_task(self):
        self.task_ = []

    def MergeFrom(self, x):
        assert x is not self
        for i in range(x.task_size()):
            self.add_task().CopyFrom(x.task(i))

    def Equals(self, x):
        if x is self:
            return 1
        if len(self.task_) != len(x.task_):
            return 0
        for e1, e2 in zip(self.task_, x.task_):
            if e1 != e2:
                return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        for p in self.task_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += 2 * len(self.task_)
        for i in range(len(self.task_)):
            n += self.task_[i].ByteSize()
        return n

    def ByteSizePartial(self):
        n = 0
        n += 2 * len(self.task_)
        for i in range(len(self.task_)):
            n += self.task_[i].ByteSizePartial()
        return n

    def Clear(self):
        self.clear_task()

    def OutputUnchecked(self, out):
        for i in range(len(self.task_)):
            out.putVarInt32(11)
            self.task_[i].OutputUnchecked(out)
            out.putVarInt32(12)

    def OutputPartial(self, out):
        for i in range(len(self.task_)):
            out.putVarInt32(11)
            self.task_[i].OutputPartial(out)
            out.putVarInt32(12)

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 11:
                self.add_task().TryMerge(d)
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        cnt = 0
        for e in self.task_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'Task%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    kTaskGroup = 1
    kTasktask_name = 2
    kTasketa_usec = 3
    kTaskurl = 4
    kTaskmethod = 5
    kTaskretry_count = 6
    kTaskHeaderGroup = 7
    kTaskHeaderkey = 8
    kTaskHeadervalue = 9
    kTaskbody_size = 10
    kTaskbody = 11
    kTaskcreation_time_usec = 12
    kTaskCronTimetableGroup = 13
    kTaskCronTimetableschedule = 14
    kTaskCronTimetabletimezone = 15
    kTaskRunLogGroup = 16
    kTaskRunLogdispatched_usec = 17
    kTaskRunLoglag_usec = 18
    kTaskRunLogelapsed_usec = 19
    kTaskRunLogresponse_code = 20
    kTaskRunLogretry_reason = 27
    kTaskdescription = 21
    kTaskpayload = 22
    kTaskretry_parameters = 23
    kTaskfirst_try_usec = 24
    kTasktag = 25
    kTaskexecution_count = 26
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode', 1: 'Task', 2: 'task_name', 3: 'eta_usec', 4: 'url', 5: 'method', 6: 'retry_count', 7: 'Header', 8: 'key', 9: 'value', 10: 'body_size', 11: 'body', 12: 'creation_time_usec', 13: 'CronTimetable', 14: 'schedule', 15: 'timezone', 16: 'RunLog', 17: 'dispatched_usec', 18: 'lag_usec', 19: 'elapsed_usec', 20: 'response_code', 21: 'description', 22: 'payload', 23: 'retry_parameters', 24: 'first_try_usec', 25: 'tag', 26: 'execution_count', 27: 'retry_reason'}, 27)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC, 1: ProtocolBuffer.Encoder.STARTGROUP, 2: ProtocolBuffer.Encoder.STRING, 3: ProtocolBuffer.Encoder.NUMERIC, 4: ProtocolBuffer.Encoder.STRING, 5: ProtocolBuffer.Encoder.NUMERIC, 6: ProtocolBuffer.Encoder.NUMERIC, 7: ProtocolBuffer.Encoder.STARTGROUP, 8: ProtocolBuffer.Encoder.STRING, 9: ProtocolBuffer.Encoder.STRING, 10: ProtocolBuffer.Encoder.NUMERIC, 11: ProtocolBuffer.Encoder.STRING, 12: ProtocolBuffer.Encoder.NUMERIC, 13: ProtocolBuffer.Encoder.STARTGROUP, 14: ProtocolBuffer.Encoder.STRING, 15: ProtocolBuffer.Encoder.STRING, 16: ProtocolBuffer.Encoder.STARTGROUP, 17: ProtocolBuffer.Encoder.NUMERIC, 18: ProtocolBuffer.Encoder.NUMERIC, 19: ProtocolBuffer.Encoder.NUMERIC, 20: ProtocolBuffer.Encoder.NUMERIC, 21: ProtocolBuffer.Encoder.STRING, 22: ProtocolBuffer.Encoder.STRING, 23: ProtocolBuffer.Encoder.STRING, 24: ProtocolBuffer.Encoder.NUMERIC, 25: ProtocolBuffer.Encoder.STRING, 26: ProtocolBuffer.Encoder.NUMERIC, 27: ProtocolBuffer.Encoder.STRING}, 27, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueQueryTasksResponse'