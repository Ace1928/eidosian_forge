from .api import Request, Response
from .types import (
class CreateTopicsRequest_v3(Request):
    API_KEY = 19
    API_VERSION = 3
    RESPONSE_TYPE = CreateTopicsResponse_v3
    SCHEMA = CreateTopicsRequest_v1.SCHEMA