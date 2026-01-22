import errno
import os
import select
import socket
import sys
import ovs.timeval
import ovs.vlog
class _SelectSelect(object):
    """ select.poll emulation by using select.select.
    Only register and poll are needed at the moment.
    """

    def __init__(self):
        self.rlist = []
        self.wlist = []
        self.xlist = []

    def register(self, fd, events):
        if isinstance(fd, socket.socket):
            fd = fd.fileno()
        if ssl and isinstance(fd, ssl.SSLSocket):
            fd = fd.fileno()
        if sys.platform != 'win32':
            assert isinstance(fd, int)
        if events & POLLIN:
            self.rlist.append(fd)
            events &= ~POLLIN
        if events & POLLOUT:
            self.wlist.append(fd)
            events &= ~POLLOUT
        if events:
            self.xlist.append(fd)

    def poll(self, timeout):
        if timeout == 0 and _using_eventlet_green_select():
            timeout = 0.1
        if sys.platform == 'win32':
            events = self.rlist + self.wlist + self.xlist
            if not events:
                return []
            if len(events) > winutils.win32event.MAXIMUM_WAIT_OBJECTS:
                raise WindowsError('Cannot handle more than maximum waitobjects\n')
            if timeout == 0.1:
                timeout = 100
            else:
                timeout = int(timeout)
            try:
                retval = winutils.win32event.WaitForMultipleObjects(events, False, timeout)
            except winutils.pywintypes.error:
                return [(0, POLLERR)]
            if retval == winutils.winerror.WAIT_TIMEOUT:
                return []
            if events[retval] in self.rlist:
                revent = POLLIN
            elif events[retval] in self.wlist:
                revent = POLLOUT
            else:
                revent = POLLERR
            return [(events[retval], revent)]
        else:
            if timeout == -1:
                timeout = None
            else:
                timeout = float(timeout) / 1000
            rlist, wlist, xlist = select.select(self.rlist, self.wlist, self.xlist, timeout)
            events_dict = {}
            for fd in rlist:
                events_dict[fd] = events_dict.get(fd, 0) | POLLIN
            for fd in wlist:
                events_dict[fd] = events_dict.get(fd, 0) | POLLOUT
            for fd in xlist:
                events_dict[fd] = events_dict.get(fd, 0) | (POLLERR | POLLHUP | POLLNVAL)
            return list(events_dict.items())