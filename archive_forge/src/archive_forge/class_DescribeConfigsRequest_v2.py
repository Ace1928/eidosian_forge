from .api import Request, Response
from .types import (
class DescribeConfigsRequest_v2(Request):
    API_KEY = 32
    API_VERSION = 2
    RESPONSE_TYPE = DescribeConfigsResponse_v2
    SCHEMA = DescribeConfigsRequest_v1.SCHEMA