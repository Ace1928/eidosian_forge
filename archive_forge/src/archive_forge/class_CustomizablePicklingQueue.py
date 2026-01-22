import copyreg
import sys
import warnings
from time import sleep
from pickle import Pickler
from pickle import HIGHEST_PROTOCOL
from io import BytesIO
from ._memmapping_reducer import get_memmapping_reducers
from ._memmapping_reducer import TemporaryResourcesManager
from ._multiprocessing_helpers import mp, assert_spawning
from multiprocessing.pool import Pool
class CustomizablePicklingQueue(object):
    """Locked Pipe implementation that uses a customizable pickler.

    This class is an alternative to the multiprocessing implementation
    of SimpleQueue in order to make it possible to pass custom
    pickling reducers, for instance to avoid memory copy when passing
    memory mapped datastructures.

    `reducers` is expected to be a dict with key / values being
    `(type, callable)` pairs where `callable` is a function that, given an
    instance of `type`, will return a tuple `(constructor, tuple_of_objects)`
    to rebuild an instance out of the pickled `tuple_of_objects` as would
    return a `__reduce__` method.

    See the standard library documentation on pickling for more details.
    """

    def __init__(self, context, reducers=None):
        self._reducers = reducers
        self._reader, self._writer = context.Pipe(duplex=False)
        self._rlock = context.Lock()
        if sys.platform == 'win32':
            self._wlock = None
        else:
            self._wlock = context.Lock()
        self._make_methods()

    def __getstate__(self):
        assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock, self._reducers)

    def __setstate__(self, state):
        self._reader, self._writer, self._rlock, self._wlock, self._reducers = state
        self._make_methods()

    def empty(self):
        return not self._reader.poll()

    def _make_methods(self):
        self._recv = recv = self._reader.recv
        racquire, rrelease = (self._rlock.acquire, self._rlock.release)

        def get():
            racquire()
            try:
                return recv()
            finally:
                rrelease()
        self.get = get
        if self._reducers:

            def send(obj):
                buffer = BytesIO()
                CustomizablePickler(buffer, self._reducers).dump(obj)
                self._writer.send_bytes(buffer.getvalue())
            self._send = send
        else:
            self._send = send = self._writer.send
        if self._wlock is None:
            self.put = send
        else:
            wlock_acquire, wlock_release = (self._wlock.acquire, self._wlock.release)

            def put(obj):
                wlock_acquire()
                try:
                    return send(obj)
                finally:
                    wlock_release()
            self.put = put