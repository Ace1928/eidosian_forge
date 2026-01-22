from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String, Bytes
class FetchRequest_v6(Request):
    """
    The body of FETCH_REQUEST_V6 is the same as FETCH_REQUEST_V5. The version number is
    bumped up to indicate that the client supports KafkaStorageException. The
    KafkaStorageException will be translated to NotLeaderForPartitionException in the
    response if version <= 5
    """
    API_KEY = 1
    API_VERSION = 6
    RESPONSE_TYPE = FetchResponse_v6
    SCHEMA = FetchRequest_v5.SCHEMA