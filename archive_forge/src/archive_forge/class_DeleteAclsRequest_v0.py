from .api import Request, Response
from .types import (
class DeleteAclsRequest_v0(Request):
    API_KEY = 31
    API_VERSION = 0
    RESPONSE_TYPE = DeleteAclsResponse_v0
    SCHEMA = Schema(('filters', Array(('resource_type', Int8), ('resource_name', String('utf-8')), ('principal', String('utf-8')), ('host', String('utf-8')), ('operation', Int8), ('permission_type', Int8))))