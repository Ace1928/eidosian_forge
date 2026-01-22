from .api import Request, Response
from .types import (
class DescribeClientQuotasResponse_v0(Request):
    API_KEY = 48
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16), ('error_message', String('utf-8')), ('entries', Array(('entity', Array(('entity_type', String('utf-8')), ('entity_name', String('utf-8')))), ('values', Array(('name', String('utf-8')), ('value', Float64))))))