import enum
import functools
import redis
from redis import exceptions as redis_exceptions
def _raise_on_closed(meth):

    @functools.wraps(meth)
    def wrapper(self, *args, **kwargs):
        if self.closed:
            raise redis_exceptions.ConnectionError('Connection has been closed')
        return meth(self, *args, **kwargs)
    return wrapper