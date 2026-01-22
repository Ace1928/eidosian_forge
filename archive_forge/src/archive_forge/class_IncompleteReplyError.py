import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
class IncompleteReplyError(DNSError):
    pass