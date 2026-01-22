from __future__ import annotations
import gevent
from gevent import select
import zmq
from zmq import Poller as _original_Poller
def _get_descriptors(self):
    """Returns three elements tuple with socket descriptors ready
        for gevent.select.select
        """
    rlist = []
    wlist = []
    xlist = []
    for socket, flags in self.sockets:
        if isinstance(socket, zmq.Socket):
            rlist.append(socket.getsockopt(zmq.FD))
            continue
        elif isinstance(socket, int):
            fd = socket
        elif hasattr(socket, 'fileno'):
            try:
                fd = int(socket.fileno())
            except Exception:
                raise ValueError('fileno() must return an valid integer fd')
        else:
            raise TypeError('Socket must be a 0MQ socket, an integer fd or have a fileno() method: %r' % socket)
        if flags & zmq.POLLIN:
            rlist.append(fd)
        if flags & zmq.POLLOUT:
            wlist.append(fd)
        if flags & zmq.POLLERR:
            xlist.append(fd)
    return (rlist, wlist, xlist)