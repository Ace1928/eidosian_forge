from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceResponse_v8(Response):
    """
    V8 bumped up to add two new fields record_errors offset list and error_message
    (See KIP-467)
    """
    API_KEY = 0
    API_VERSION = 8
    SCHEMA = Schema(('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('error_code', Int16), ('offset', Int64), ('timestamp', Int64), ('log_start_offset', Int64)), ('record_errors', Array(('batch_index', Int32), ('batch_index_error_message', String('utf-8')))), ('error_message', String('utf-8'))))), ('throttle_time_ms', Int32))