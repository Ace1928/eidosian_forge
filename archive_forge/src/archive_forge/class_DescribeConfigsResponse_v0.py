from .api import Request, Response
from .types import (
class DescribeConfigsResponse_v0(Response):
    API_KEY = 32
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('resources', Array(('error_code', Int16), ('error_message', String('utf-8')), ('resource_type', Int8), ('resource_name', String('utf-8')), ('config_entries', Array(('config_names', String('utf-8')), ('config_value', String('utf-8')), ('read_only', Boolean), ('is_default', Boolean), ('is_sensitive', Boolean))))))