from .api import Request, Response
from .types import (
class ListGroupsResponse_v1(Response):
    API_KEY = 16
    API_VERSION = 1
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16), ('groups', Array(('group', String('utf-8')), ('protocol_type', String('utf-8')))))