from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetCommitResponse_v1(Response):
    API_KEY = 8
    API_VERSION = 1
    SCHEMA = OffsetCommitResponse_v0.SCHEMA