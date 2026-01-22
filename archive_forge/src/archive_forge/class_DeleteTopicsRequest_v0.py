from .api import Request, Response
from .types import (
class DeleteTopicsRequest_v0(Request):
    API_KEY = 20
    API_VERSION = 0
    RESPONSE_TYPE = DeleteTopicsResponse_v0
    SCHEMA = Schema(('topics', Array(String('utf-8'))), ('timeout', Int32))