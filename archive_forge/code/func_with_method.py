import inspect
import wsme.api
@classmethod
def with_method(cls, method, *args, **kwargs):
    kwargs['method'] = method
    return cls(*args, **kwargs)