from .api import Request, Response
from .types import Int16, Int32, Int64, Schema, String, Array, Boolean
class TxnOffsetCommitResponse_v0(Response):
    API_KEY = 28
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('errors', Array(('topic', String('utf-8')), ('partition_errors', Array(('partition', Int32), ('error_code', Int16))))))