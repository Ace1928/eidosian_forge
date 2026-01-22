from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class SyncGroupRequest_v1(Request):
    API_KEY = 14
    API_VERSION = 1
    RESPONSE_TYPE = SyncGroupResponse_v1
    SCHEMA = SyncGroupRequest_v0.SCHEMA