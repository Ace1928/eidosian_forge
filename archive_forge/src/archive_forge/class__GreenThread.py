import eventlet
from eventlet.green import thread
from eventlet.green import time
from eventlet.support import greenlets as greenlet
class _GreenThread:
    """Wrapper for GreenThread objects to provide Thread-like attributes
    and methods"""

    def __init__(self, g):
        global _count
        self._g = g
        self._name = 'GreenThread-%d' % _count
        _count += 1

    def __repr__(self):
        return '<_GreenThread(%s, %r)>' % (self._name, self._g)

    def join(self, timeout=None):
        return self._g.wait()

    def getName(self):
        return self._name
    get_name = getName

    def setName(self, name):
        self._name = str(name)
    set_name = setName
    name = property(getName, setName)
    ident = property(lambda self: id(self._g))

    def isAlive(self):
        return True
    is_alive = isAlive
    daemon = property(lambda self: True)

    def isDaemon(self):
        return self.daemon
    is_daemon = isDaemon