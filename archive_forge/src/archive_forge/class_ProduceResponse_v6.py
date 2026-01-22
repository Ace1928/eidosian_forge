from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceResponse_v6(Response):
    """
    The version number is bumped to indicate that on quota violation brokers send out
    responses before throttling.
    """
    API_KEY = 0
    API_VERSION = 6
    SCHEMA = ProduceResponse_v5.SCHEMA