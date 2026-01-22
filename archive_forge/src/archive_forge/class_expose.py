import inspect
import wsme.api
class expose(object):

    def __init__(self, *args, **kwargs):
        self.signature = wsme.api.signature(*args, **kwargs)

    def __call__(self, func):
        return self.signature(func)

    @classmethod
    def with_method(cls, method, *args, **kwargs):
        kwargs['method'] = method
        return cls(*args, **kwargs)

    @classmethod
    def get(cls, *args, **kwargs):
        return cls.with_method('GET', *args, **kwargs)

    @classmethod
    def post(cls, *args, **kwargs):
        return cls.with_method('POST', *args, **kwargs)

    @classmethod
    def put(cls, *args, **kwargs):
        return cls.with_method('PUT', *args, **kwargs)

    @classmethod
    def delete(cls, *args, **kwargs):
        return cls.with_method('DELETE', *args, **kwargs)