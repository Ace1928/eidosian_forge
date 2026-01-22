import multiprocessing
import requests
from . import thread
from .._compat import queue
class ThreadException(ThreadProxy):
    """A wrapper around an exception raised during a request.

    This will proxy most attribute access actions to the exception object. For
    example, if you wanted the message from the exception, you might do:

    .. code-block:: python

        thread_exc = pool.get_exception()
        msg = thread_exc.message

    """
    proxied_attr = 'exception'
    attrs = frozenset(['request_kwargs', 'exception'])

    def __init__(self, request_kwargs, exception):
        self.request_kwargs = request_kwargs
        self.exception = exception