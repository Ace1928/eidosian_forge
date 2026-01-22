from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetResponse_v5(Response):
    """
    adds a new error code, OFFSET_NOT_AVAILABLE
    """
    API_KEY = 2
    API_VERSION = 5
    SCHEMA = OffsetResponse_v4.SCHEMA