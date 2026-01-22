from concurrent.futures import _base
import itertools
import queue
import threading
import types
import weakref
import os
class _WorkItem(object):

    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if not self.future.set_running_or_notify_cancel():
            return
        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:
            self.future.set_exception(exc)
            self = None
        else:
            self.future.set_result(result)
    __class_getitem__ = classmethod(types.GenericAlias)