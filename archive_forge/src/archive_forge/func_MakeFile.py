import socket
import _pyio as io
def MakeFile(sock, mode='r', bufsize=io.DEFAULT_BUFFER_SIZE):
    """File object attached to a socket object."""
    cls = StreamReader if 'r' in mode else StreamWriter
    return cls(sock, mode, bufsize)