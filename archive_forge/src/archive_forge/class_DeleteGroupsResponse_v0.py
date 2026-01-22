from .api import Request, Response
from .types import (
class DeleteGroupsResponse_v0(Response):
    API_KEY = 42
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('results', Array(('group_id', String('utf-8')), ('error_code', Int16))))