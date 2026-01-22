from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceRequest(Request):
    API_KEY = 0

    def expect_response(self):
        if self.required_acks == 0:
            return False
        return True