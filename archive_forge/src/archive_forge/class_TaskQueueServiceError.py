from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb import *
import googlecloudsdk.third_party.appengine.datastore.datastore_v3_pb
from googlecloudsdk.third_party.appengine.proto.message_set import MessageSet
class TaskQueueServiceError(ProtocolBuffer.ProtocolMessage):
    OK = 0
    UNKNOWN_QUEUE = 1
    TRANSIENT_ERROR = 2
    INTERNAL_ERROR = 3
    TASK_TOO_LARGE = 4
    INVALID_TASK_NAME = 5
    INVALID_QUEUE_NAME = 6
    INVALID_URL = 7
    INVALID_QUEUE_RATE = 8
    PERMISSION_DENIED = 9
    TASK_ALREADY_EXISTS = 10
    TOMBSTONED_TASK = 11
    INVALID_ETA = 12
    INVALID_REQUEST = 13
    UNKNOWN_TASK = 14
    TOMBSTONED_QUEUE = 15
    DUPLICATE_TASK_NAME = 16
    SKIPPED = 17
    TOO_MANY_TASKS = 18
    INVALID_PAYLOAD = 19
    INVALID_RETRY_PARAMETERS = 20
    INVALID_QUEUE_MODE = 21
    ACL_LOOKUP_ERROR = 22
    TRANSACTIONAL_REQUEST_TOO_LARGE = 23
    INCORRECT_CREATOR_NAME = 24
    TASK_LEASE_EXPIRED = 25
    QUEUE_PAUSED = 26
    INVALID_TAG = 27
    INVALID_LOGGING_CONFIG = 28
    DATASTORE_ERROR = 10000
    _ErrorCode_NAMES = {0: 'OK', 1: 'UNKNOWN_QUEUE', 2: 'TRANSIENT_ERROR', 3: 'INTERNAL_ERROR', 4: 'TASK_TOO_LARGE', 5: 'INVALID_TASK_NAME', 6: 'INVALID_QUEUE_NAME', 7: 'INVALID_URL', 8: 'INVALID_QUEUE_RATE', 9: 'PERMISSION_DENIED', 10: 'TASK_ALREADY_EXISTS', 11: 'TOMBSTONED_TASK', 12: 'INVALID_ETA', 13: 'INVALID_REQUEST', 14: 'UNKNOWN_TASK', 15: 'TOMBSTONED_QUEUE', 16: 'DUPLICATE_TASK_NAME', 17: 'SKIPPED', 18: 'TOO_MANY_TASKS', 19: 'INVALID_PAYLOAD', 20: 'INVALID_RETRY_PARAMETERS', 21: 'INVALID_QUEUE_MODE', 22: 'ACL_LOOKUP_ERROR', 23: 'TRANSACTIONAL_REQUEST_TOO_LARGE', 24: 'INCORRECT_CREATOR_NAME', 25: 'TASK_LEASE_EXPIRED', 26: 'QUEUE_PAUSED', 27: 'INVALID_TAG', 28: 'INVALID_LOGGING_CONFIG', 10000: 'DATASTORE_ERROR'}

    def ErrorCode_Name(cls, x):
        return cls._ErrorCode_NAMES.get(x, '')
    ErrorCode_Name = classmethod(ErrorCode_Name)

    def __init__(self, contents=None):
        pass
        if contents is not None:
            self.MergeFromString(contents)

    def MergeFrom(self, x):
        assert x is not self

    def Equals(self, x):
        if x is self:
            return 1
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        return initialized

    def ByteSize(self):
        n = 0
        return n

    def ByteSizePartial(self):
        n = 0
        return n

    def Clear(self):
        pass

    def OutputUnchecked(self, out):
        pass

    def OutputPartial(self, out):
        pass

    def TryMerge(self, d):
        while d.avail() > 0:
            tt = d.getVarInt32()
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        return res

    def _BuildTagLookupTable(sparse, maxtag, default=None):
        return tuple([sparse.get(i, default) for i in range(0, 1 + maxtag)])
    _TEXT = _BuildTagLookupTable({0: 'ErrorCode'}, 0)
    _TYPES = _BuildTagLookupTable({0: ProtocolBuffer.Encoder.NUMERIC}, 0, ProtocolBuffer.Encoder.MAX_TYPE)
    _STYLE = ''
    _STYLE_CONTENT_TYPE = ''
    _PROTO_DESCRIPTOR_NAME = 'apphosting.TaskQueueServiceError'