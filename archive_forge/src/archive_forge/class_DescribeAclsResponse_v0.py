from .api import Request, Response
from .types import (
class DescribeAclsResponse_v0(Response):
    API_KEY = 29
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16), ('error_message', String('utf-8')), ('resources', Array(('resource_type', Int8), ('resource_name', String('utf-8')), ('acls', Array(('principal', String('utf-8')), ('host', String('utf-8')), ('operation', Int8), ('permission_type', Int8))))))