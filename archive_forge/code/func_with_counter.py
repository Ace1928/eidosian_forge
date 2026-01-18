from functools import wraps
import weakref
import abc
import warnings
from ..data_sparsifier import BaseDataSparsifier
def with_counter(method):
    if getattr(method, '_with_counter', False):
        return method
    instance_ref = weakref.ref(method.__self__)
    func = method.__func__
    cls = instance_ref().__class__
    del method

    @wraps(func)
    def wrapper(*args, **kwargs):
        instance = instance_ref()
        instance._step_count += 1
        wrapped = func.__get__(instance, cls)
        return wrapped(*args, **kwargs)
    wrapper._with_counter = True
    return wrapper