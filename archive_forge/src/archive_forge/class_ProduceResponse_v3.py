from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceResponse_v3(Response):
    API_KEY = 0
    API_VERSION = 3
    SCHEMA = ProduceResponse_v2.SCHEMA