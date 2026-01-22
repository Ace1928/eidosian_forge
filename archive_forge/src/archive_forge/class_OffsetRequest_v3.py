from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetRequest_v3(Request):
    API_KEY = 2
    API_VERSION = 3
    RESPONSE_TYPE = OffsetResponse_v3
    SCHEMA = OffsetRequest_v2.SCHEMA
    DEFAULTS = {'replica_id': -1}