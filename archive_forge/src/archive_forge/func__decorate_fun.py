import functools
import warnings
def _decorate_fun(self, fun):
    """Decorate function fun"""
    msg = 'Function %s is deprecated' % fun.__name__
    if self.extra:
        msg += '; %s' % self.extra

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        warnings.warn(msg, category=FutureWarning)
        return fun(*args, **kwargs)
    wrapped.__wrapped__ = fun
    return wrapped