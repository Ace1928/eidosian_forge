import threading
import sys
from paste.util import filemixin
class PrintCatcher(filemixin.FileMixin):

    def __init__(self, default=None, factory=None, paramwriter=None, leave_stdout=False):
        assert len(filter(lambda x: x is not None, [default, factory, paramwriter])) <= 1, 'You can only provide one of default, factory, or paramwriter'
        if leave_stdout:
            assert not default, 'You cannot pass in both default (%r) and leave_stdout=True' % default
            default = sys.stdout
        if default:
            self._defaultfunc = self._writedefault
        elif factory:
            self._defaultfunc = self._writefactory
        elif paramwriter:
            self._defaultfunc = self._writeparam
        else:
            self._defaultfunc = self._writeerror
        self._default = default
        self._factory = factory
        self._paramwriter = paramwriter
        self._catchers = {}

    def write(self, v, currentThread=threading.current_thread):
        name = currentThread().name
        catchers = self._catchers
        if not catchers.has_key(name):
            self._defaultfunc(name, v)
        else:
            catcher = catchers[name]
            catcher.write(v)

    def seek(self, *args):
        name = threading.current_thread().name
        catchers = self._catchers
        if not name in catchers:
            self._default.seek(*args)
        else:
            catchers[name].seek(*args)

    def read(self, *args):
        name = threading.current_thread().name
        catchers = self._catchers
        if not name in catchers:
            self._default.read(*args)
        else:
            catchers[name].read(*args)

    def _writedefault(self, name, v):
        self._default.write(v)

    def _writefactory(self, name, v):
        self._factory(name).write(v)

    def _writeparam(self, name, v):
        self._paramwriter(name, v)

    def _writeerror(self, name, v):
        assert False, 'There is no PrintCatcher output stream for the thread %r' % name

    def register(self, catcher, name=None, currentThread=threading.current_thread):
        if name is None:
            name = currentThread().name
        self._catchers[name] = catcher

    def deregister(self, name=None, currentThread=threading.current_thread):
        if name is None:
            name = currentThread().name
        assert self._catchers.has_key(name), 'There is no PrintCatcher catcher for the thread %r' % name
        del self._catchers[name]