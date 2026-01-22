import functools
import warnings
def _decorate_class(self, cls):
    msg = 'Class %s is deprecated' % cls.__name__
    if self.extra:
        msg += '; %s' % self.extra
    new = cls.__new__

    def wrapped(cls, *args, **kwargs):
        warnings.warn(msg, category=FutureWarning)
        if new is object.__new__:
            return object.__new__(cls)
        return new(cls, *args, **kwargs)
    cls.__new__ = wrapped
    wrapped.__name__ = '__new__'
    wrapped.deprecated_original = new
    return cls