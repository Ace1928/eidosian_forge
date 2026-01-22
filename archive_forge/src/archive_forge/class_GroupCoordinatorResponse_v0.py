from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class GroupCoordinatorResponse_v0(Response):
    API_KEY = 10
    API_VERSION = 0
    SCHEMA = Schema(('error_code', Int16), ('coordinator_id', Int32), ('host', String('utf-8')), ('port', Int32))