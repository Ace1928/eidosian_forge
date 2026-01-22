from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceResponse_v5(Response):
    API_KEY = 0
    API_VERSION = 5
    SCHEMA = Schema(('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('error_code', Int16), ('offset', Int64), ('timestamp', Int64), ('log_start_offset', Int64))))), ('throttle_time_ms', Int32))