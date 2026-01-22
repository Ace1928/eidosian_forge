from functools import wraps
class XLock(threadingmodule._RLock):

    def __reduce__(self):
        return (unpickle_lock, ())