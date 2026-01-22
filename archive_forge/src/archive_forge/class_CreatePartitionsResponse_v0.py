from .api import Request, Response
from .types import (
class CreatePartitionsResponse_v0(Response):
    API_KEY = 37
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('topic_errors', Array(('topic', String('utf-8')), ('error_code', Int16), ('error_message', String('utf-8')))))