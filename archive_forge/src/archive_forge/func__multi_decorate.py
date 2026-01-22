from functools import partial, update_wrapper, wraps
from asgiref.sync import iscoroutinefunction
def _multi_decorate(decorators, method):
    """
    Decorate `method` with one or more function decorators. `decorators` can be
    a single decorator or an iterable of decorators.
    """
    if hasattr(decorators, '__iter__'):
        decorators = decorators[::-1]
    else:
        decorators = [decorators]

    def _wrapper(self, *args, **kwargs):
        bound_method = wraps(method)(partial(method.__get__(self, type(self))))
        for dec in decorators:
            bound_method = dec(bound_method)
        return bound_method(*args, **kwargs)
    for dec in decorators:
        _update_method_wrapper(_wrapper, dec)
    update_wrapper(_wrapper, method)
    return _wrapper