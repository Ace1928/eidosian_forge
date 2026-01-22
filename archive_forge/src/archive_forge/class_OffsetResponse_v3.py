from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class OffsetResponse_v3(Response):
    """
    on quota violation, brokers send out responses before throttling
    """
    API_KEY = 2
    API_VERSION = 3
    SCHEMA = OffsetResponse_v2.SCHEMA