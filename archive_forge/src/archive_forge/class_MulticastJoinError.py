import socket
from incremental import Version
from twisted.python import deprecate
class MulticastJoinError(Exception):
    """
    An attempt to join a multicast group failed.
    """