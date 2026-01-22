from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class GroupCoordinatorRequest_v0(Request):
    API_KEY = 10
    API_VERSION = 0
    RESPONSE_TYPE = GroupCoordinatorResponse_v0
    SCHEMA = Schema(('consumer_group', String('utf-8')))