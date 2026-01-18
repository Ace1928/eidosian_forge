import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def voidcmd(self, cmd):
    """Send a command and expect a response beginning with '2'."""
    self.putcmd(cmd)
    return self.voidresp()