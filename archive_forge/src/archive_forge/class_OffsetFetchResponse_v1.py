from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetFetchResponse_v1(Response):
    API_KEY = 9
    API_VERSION = 1
    SCHEMA = OffsetFetchResponse_v0.SCHEMA