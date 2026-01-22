import errno
import re
import socket
import sys
def _getresp(self):
    resp, o = self._getline()
    if self._debugging > 1:
        print('*resp*', repr(resp))
    if not resp.startswith(b'+'):
        raise error_proto(resp)
    return resp