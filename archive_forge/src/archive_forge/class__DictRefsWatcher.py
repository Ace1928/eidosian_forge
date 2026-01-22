import os
from dulwich.errors import (
from dulwich.objects import (
from dulwich.file import (
class _DictRefsWatcher(object):

    def __init__(self, refs):
        self._refs = refs

    def __enter__(self):
        from queue import Queue
        self.queue = Queue()
        self._refs._watchers.add(self)
        return self

    def __next__(self):
        return self.queue.get()

    def _notify(self, entry):
        self.queue.put_nowait(entry)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._refs._watchers.remove(self)
        return False