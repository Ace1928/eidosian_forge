import abc
from .struct import Struct
from .types import Int16, Int32, String, Schema, Array, TaggedFields
class RequestHeaderV2(Struct):
    SCHEMA = Schema(('api_key', Int16), ('api_version', Int16), ('correlation_id', Int32), ('client_id', String('utf-8')), ('tags', TaggedFields))

    def __init__(self, request, correlation_id=0, client_id='kafka-python', tags=None):
        super(RequestHeaderV2, self).__init__(request.API_KEY, request.API_VERSION, correlation_id, client_id, tags or {})