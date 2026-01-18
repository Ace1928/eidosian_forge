import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def write_fully(fd, buf):
    """Returns an (error, bytes_written) tuple where 'error' is 0 on success,
    otherwise a positive errno value, and 'bytes_written' is the number of
    bytes that were written before the error occurred.  'error' is 0 if and
    only if 'bytes_written' is len(buf)."""
    bytes_written = 0
    if len(buf) == 0:
        return (0, 0)
    if not isinstance(buf, bytes):
        buf = bytes(buf, 'utf-8')
    while True:
        try:
            retval = os.write(fd, buf)
            assert retval >= 0
            if retval == len(buf):
                return (0, bytes_written + len(buf))
            elif retval == 0:
                vlog.warn('write returned 0')
                return (errno.EPROTO, bytes_written)
            else:
                bytes_written += retval
                buf = buf[:retval]
        except OSError as e:
            return (e.errno, bytes_written)