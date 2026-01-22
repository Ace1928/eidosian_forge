from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class LeaveGroupRequest_v0(Request):
    API_KEY = 13
    API_VERSION = 0
    RESPONSE_TYPE = LeaveGroupResponse_v0
    SCHEMA = Schema(('group', String('utf-8')), ('member_id', String('utf-8')))