from .api import Request, Response
from .types import Int8, Int16, Int32, Schema, String
class FindCoordinatorResponse_v1(Response):
    API_KEY = 10
    API_VERSION = 1
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16), ('error_message', String('utf-8')), ('coordinator_id', Int32), ('host', String('utf-8')), ('port', Int32))