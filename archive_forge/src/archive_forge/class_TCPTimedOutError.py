import socket
from incremental import Version
from twisted.python import deprecate
class TCPTimedOutError(ConnectError):
    __doc__ = MESSAGE = 'TCP connection timed out'