import collections.abc
import pycares
import select
import socket
import sys
def wait_channel(channel):
    while True:
        read_fds, write_fds = channel.getsock()
        if not read_fds and (not write_fds):
            break
        timeout = channel.timeout()
        if not timeout:
            channel.process_fd(pycares.ARES_SOCKET_BAD, pycares.ARES_SOCKET_BAD)
            continue
        rlist, wlist, xlist = select.select(read_fds, write_fds, [], timeout)
        for fd in rlist:
            channel.process_fd(fd, pycares.ARES_SOCKET_BAD)
        for fd in wlist:
            channel.process_fd(pycares.ARES_SOCKET_BAD, fd)