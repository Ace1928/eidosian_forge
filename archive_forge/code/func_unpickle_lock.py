from functools import wraps
def unpickle_lock():
    if threadingmodule is not None:
        return XLock()
    else:
        return DummyLock()