from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def _call_and_read_response(self, method, args, body=None, readv_body=None, body_stream=None, expect_response_body=True):
    request = _SmartClientRequest(self, method, args, body=body, readv_body=readv_body, body_stream=body_stream, expect_response_body=expect_response_body)
    return request.call_and_read_response()