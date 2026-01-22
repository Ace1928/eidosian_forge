from .api import Request, Response
from .types import (
class CreateTopicsResponse_v0(Response):
    API_KEY = 19
    API_VERSION = 0
    SCHEMA = Schema(('topic_errors', Array(('topic', String('utf-8')), ('error_code', Int16))))