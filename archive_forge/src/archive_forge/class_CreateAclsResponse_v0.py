from .api import Request, Response
from .types import (
class CreateAclsResponse_v0(Response):
    API_KEY = 30
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('creation_responses', Array(('error_code', Int16), ('error_message', String('utf-8')))))