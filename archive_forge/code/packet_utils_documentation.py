import array
import socket
import struct
from os_ken.lib import addrconv

    Fletcher Checksum -- Refer to RFC1008

    calling with offset == _FLETCHER_CHECKSUM_VALIDATE will validate the
    checksum without modifying the buffer; a valid checksum returns 0.
    