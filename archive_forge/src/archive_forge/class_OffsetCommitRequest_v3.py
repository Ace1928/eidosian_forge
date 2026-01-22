from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetCommitRequest_v3(Request):
    API_KEY = 8
    API_VERSION = 3
    RESPONSE_TYPE = OffsetCommitResponse_v3
    SCHEMA = OffsetCommitRequest_v2.SCHEMA