from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class JoinGroupRequest_v2(Request):
    API_KEY = 11
    API_VERSION = 2
    RESPONSE_TYPE = JoinGroupResponse_v2
    SCHEMA = JoinGroupRequest_v1.SCHEMA
    UNKNOWN_MEMBER_ID = ''