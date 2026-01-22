import array
import os
import socket
from warnings import warn
Make a list of FileDescriptor from received file descriptors

        ancdata is a list of ancillary data tuples as returned by socket.recvmsg()
        