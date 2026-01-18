import array
import ctypes.wintypes
import platform
import struct
from paramiko.common import zero_byte
from paramiko.util import b
import _thread as thread
from . import _winapi

    Mock "connection" to an agent which roughly approximates the behavior of
    a unix local-domain socket (as used by Agent).  Requests are sent to the
    pageant daemon via special Windows magick, and responses are buffered back
    for subsequent reads.
    