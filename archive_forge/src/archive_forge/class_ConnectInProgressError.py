import socket
from incremental import Version
from twisted.python import deprecate
class ConnectInProgressError(Exception):
    """A connect operation was started and isn't done yet."""