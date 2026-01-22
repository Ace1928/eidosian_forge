from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceResponse_v7(Response):
    """
    V7 bumped up to indicate ZStandard capability. (see KIP-110)
    """
    API_KEY = 0
    API_VERSION = 7
    SCHEMA = ProduceResponse_v6.SCHEMA