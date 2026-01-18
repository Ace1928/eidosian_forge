import functools
import inspect
import logging
import traceback
import wsme.exc
import wsme.types
from wsme import utils
def wrapfunc(f):

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    wrapper._wsme_original_func = f
    return wrapper