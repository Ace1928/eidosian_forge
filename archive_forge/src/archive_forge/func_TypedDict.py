import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
@_ensure_subclassable(lambda bases: (_TypedDict,))
def TypedDict(typename, fields=_marker, /, *, total=True, closed=False, **kwargs):
    """A simple typed namespace. At runtime it is equivalent to a plain dict.

        TypedDict creates a dictionary type such that a type checker will expect all
        instances to have a certain set of keys, where each key is
        associated with a value of a consistent type. This expectation
        is not checked at runtime.

        Usage::

            class Point2D(TypedDict):
                x: int
                y: int
                label: str

            a: Point2D = {'x': 1, 'y': 2, 'label': 'good'}  # OK
            b: Point2D = {'z': 3, 'label': 'bad'}           # Fails type check

            assert Point2D(x=1, y=2, label='first') == dict(x=1, y=2, label='first')

        The type info can be accessed via the Point2D.__annotations__ dict, and
        the Point2D.__required_keys__ and Point2D.__optional_keys__ frozensets.
        TypedDict supports an additional equivalent form::

            Point2D = TypedDict('Point2D', {'x': int, 'y': int, 'label': str})

        By default, all keys must be present in a TypedDict. It is possible
        to override this by specifying totality::

            class Point2D(TypedDict, total=False):
                x: int
                y: int

        This means that a Point2D TypedDict can have any of the keys omitted. A type
        checker is only expected to support a literal False or True as the value of
        the total argument. True is the default, and makes all items defined in the
        class body be required.

        The Required and NotRequired special forms can also be used to mark
        individual keys as being required or not required::

            class Point2D(TypedDict):
                x: int  # the "x" key must always be present (Required is the default)
                y: NotRequired[int]  # the "y" key can be omitted

        See PEP 655 for more details on Required and NotRequired.
        """
    if fields is _marker or fields is None:
        if fields is _marker:
            deprecated_thing = "Failing to pass a value for the 'fields' parameter"
        else:
            deprecated_thing = "Passing `None` as the 'fields' parameter"
        example = f'`{typename} = TypedDict({typename!r}, {{}})`'
        deprecation_msg = f'{deprecated_thing} is deprecated and will be disallowed in Python 3.15. To create a TypedDict class with 0 fields using the functional syntax, pass an empty dictionary, e.g. ' + example + '.'
        warnings.warn(deprecation_msg, DeprecationWarning, stacklevel=2)
        if closed is not False and closed is not True:
            kwargs['closed'] = closed
            closed = False
        fields = kwargs
    elif kwargs:
        raise TypeError('TypedDict takes either a dict or keyword arguments, but not both')
    if kwargs:
        if sys.version_info >= (3, 13):
            raise TypeError('TypedDict takes no keyword arguments')
        warnings.warn('The kwargs-based syntax for TypedDict definitions is deprecated in Python 3.11, will be removed in Python 3.13, and may not be understood by third-party type checkers.', DeprecationWarning, stacklevel=2)
    ns = {'__annotations__': dict(fields)}
    module = _caller()
    if module is not None:
        ns['__module__'] = module
    td = _TypedDictMeta(typename, (), ns, total=total, closed=closed)
    td.__orig_bases__ = (TypedDict,)
    return td