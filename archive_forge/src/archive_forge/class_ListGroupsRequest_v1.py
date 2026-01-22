from .api import Request, Response
from .types import (
class ListGroupsRequest_v1(Request):
    API_KEY = 16
    API_VERSION = 1
    RESPONSE_TYPE = ListGroupsResponse_v1
    SCHEMA = ListGroupsRequest_v0.SCHEMA