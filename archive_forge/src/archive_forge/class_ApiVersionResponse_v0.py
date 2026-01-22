from .api import Request, Response
from .types import (
class ApiVersionResponse_v0(Response):
    API_KEY = 18
    API_VERSION = 0
    SCHEMA = Schema(('error_code', Int16), ('api_versions', Array(('api_key', Int16), ('min_version', Int16), ('max_version', Int16))))