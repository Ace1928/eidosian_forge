import abc
import collections
import collections.abc
import operator
import sys
import typing
def _typeddict_new(*args, total=True, **kwargs):
    if not args:
        raise TypeError('TypedDict.__new__(): not enough arguments')
    _, args = (args[0], args[1:])
    if args:
        typename, args = (args[0], args[1:])
    elif '_typename' in kwargs:
        typename = kwargs.pop('_typename')
        import warnings
        warnings.warn("Passing '_typename' as keyword argument is deprecated", DeprecationWarning, stacklevel=2)
    else:
        raise TypeError("TypedDict.__new__() missing 1 required positional argument: '_typename'")
    if args:
        try:
            fields, = args
        except ValueError:
            raise TypeError(f'TypedDict.__new__() takes from 2 to 3 positional arguments but {len(args) + 2} were given')
    elif '_fields' in kwargs and len(kwargs) == 1:
        fields = kwargs.pop('_fields')
        import warnings
        warnings.warn("Passing '_fields' as keyword argument is deprecated", DeprecationWarning, stacklevel=2)
    else:
        fields = None
    if fields is None:
        fields = kwargs
    elif kwargs:
        raise TypeError('TypedDict takes either a dict or keyword arguments, but not both')
    ns = {'__annotations__': dict(fields)}
    try:
        ns['__module__'] = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass
    return _TypedDictMeta(typename, (), ns, total=total)