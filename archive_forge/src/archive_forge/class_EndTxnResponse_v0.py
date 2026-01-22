from .api import Request, Response
from .types import Int16, Int32, Int64, Schema, String, Array, Boolean
class EndTxnResponse_v0(Response):
    API_KEY = 26
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16))