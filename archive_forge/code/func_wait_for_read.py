import errno
import select
import sys
from functools import partial
def wait_for_read(sock, timeout=None):
    """Waits for reading to be available on a given socket.
    Returns True if the socket is readable, or False if the timeout expired.
    """
    return wait_for_socket(sock, read=True, timeout=timeout)