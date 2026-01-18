import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def voidresp(self):
    """Expect a response beginning with '2'."""
    resp = self.getresp()
    if resp[:1] != '2':
        raise error_reply(resp)
    return resp