from .api import Request, Response
from .types import (
class SaslAuthenticateRequest_v0(Request):
    API_KEY = 36
    API_VERSION = 0
    RESPONSE_TYPE = SaslAuthenticateResponse_v0
    SCHEMA = Schema(('sasl_auth_bytes', Bytes))