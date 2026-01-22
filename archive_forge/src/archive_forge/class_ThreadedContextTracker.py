from threading import local
from typing import Dict, Type
class ThreadedContextTracker:

    def __init__(self):
        self.storage = local()

    def currentContext(self):
        try:
            return self.storage.ct
        except AttributeError:
            ct = self.storage.ct = ContextTracker()
            return ct

    def callWithContext(self, ctx, func, *args, **kw):
        return self.currentContext().callWithContext(ctx, func, *args, **kw)

    def getContext(self, key, default=None):
        return self.currentContext().getContext(key, default)