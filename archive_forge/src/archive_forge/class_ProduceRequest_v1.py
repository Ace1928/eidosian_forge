from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceRequest_v1(ProduceRequest):
    API_VERSION = 1
    RESPONSE_TYPE = ProduceResponse_v1
    SCHEMA = ProduceRequest_v0.SCHEMA