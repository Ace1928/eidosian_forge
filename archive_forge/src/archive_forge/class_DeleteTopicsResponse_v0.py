from .api import Request, Response
from .types import (
class DeleteTopicsResponse_v0(Response):
    API_KEY = 20
    API_VERSION = 0
    SCHEMA = Schema(('topic_error_codes', Array(('topic', String('utf-8')), ('error_code', Int16))))