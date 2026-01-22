from .api import Request, Response
from .types import (
class DeleteGroupsRequest_v0(Request):
    API_KEY = 42
    API_VERSION = 0
    RESPONSE_TYPE = DeleteGroupsResponse_v0
    SCHEMA = Schema(('groups_names', Array(String('utf-8'))))