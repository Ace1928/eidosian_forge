from .api import Request, Response
from .types import (
class ListGroupsResponse_v0(Response):
    API_KEY = 16
    API_VERSION = 0
    SCHEMA = Schema(('error_code', Int16), ('groups', Array(('group', String('utf-8')), ('protocol_type', String('utf-8')))))