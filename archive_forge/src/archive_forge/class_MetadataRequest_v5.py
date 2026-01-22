from .api import Request, Response
from .types import Array, Boolean, Int16, Int32, Schema, String
class MetadataRequest_v5(Request):
    """
    The v5 metadata request is the same as v4.
    An additional field for offline_replicas has been added to the v5 metadata response
    """
    API_KEY = 3
    API_VERSION = 5
    RESPONSE_TYPE = MetadataResponse_v5
    SCHEMA = MetadataRequest_v4.SCHEMA
    ALL_TOPICS = -1
    NO_TOPICS = None