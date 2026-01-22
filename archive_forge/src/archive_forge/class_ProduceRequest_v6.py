from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceRequest_v6(ProduceRequest):
    """
    The version number is bumped to indicate that on quota violation brokers send out
    responses before throttling.
    """
    API_VERSION = 6
    RESPONSE_TYPE = ProduceResponse_v6
    SCHEMA = ProduceRequest_v5.SCHEMA