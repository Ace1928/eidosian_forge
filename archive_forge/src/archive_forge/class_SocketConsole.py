from code import InteractiveConsole
import errno
import socket
import sys
import eventlet
from eventlet import hubs
from eventlet.support import greenlets, get_errno
class SocketConsole(greenlets.greenlet):

    def __init__(self, desc, hostport, locals):
        self.hostport = hostport
        self.locals = locals
        self.desc = FileProxy(desc)
        greenlets.greenlet.__init__(self)

    def run(self):
        try:
            console = InteractiveConsole(self.locals)
            console.interact()
        finally:
            self.switch_out()
            self.finalize()

    def switch(self, *args, **kw):
        self.saved = (sys.stdin, sys.stderr, sys.stdout)
        sys.stdin = sys.stdout = sys.stderr = self.desc
        greenlets.greenlet.switch(self, *args, **kw)

    def switch_out(self):
        sys.stdin, sys.stderr, sys.stdout = self.saved

    def finalize(self):
        self.desc = None
        if len(self.hostport) >= 2:
            host = self.hostport[0]
            port = self.hostport[1]
            print('backdoor closed to %s:%s' % (host, port))
        else:
            print('backdoor closed')