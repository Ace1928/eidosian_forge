from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
class SocketType(IntEnum):
    """zmq socket types

    .. versionadded:: 23
    """
    PAIR = 0
    PUB = 1
    SUB = 2
    REQ = 3
    REP = 4
    DEALER = 5
    ROUTER = 6
    PULL = 7
    PUSH = 8
    XPUB = 9
    XSUB = 10
    STREAM = 11
    XREQ = DEALER
    XREP = ROUTER
    SERVER = 12
    CLIENT = 13
    RADIO = 14
    DISH = 15
    GATHER = 16
    SCATTER = 17
    DGRAM = 18
    PEER = 19
    CHANNEL = 20