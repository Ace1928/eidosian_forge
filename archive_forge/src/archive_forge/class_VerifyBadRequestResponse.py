import inspect
import json
import sys
import traceback
class VerifyBadRequestResponse(ExpectedError):
    """
    Verifies that the test target returned a 400 Bad Request response
    containing a an error message.
    """
    cid = 'verify-bad-request-response'
    msg = 'OP error'

    def _func(self, conv):
        _response = conv.last_response
        res = {}
        if _response.status_code == 400:
            pass
        else:
            self._message = 'Expected a 400 error message'
            self._status = ERROR
        return res