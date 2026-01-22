from functools import reduce
class getitem_property:
    """Attribute -> dict key descriptor.

    The target object must support ``__getitem__``,
    and optionally ``__setitem__``.

    Example:
        >>> from collections import defaultdict

        >>> class Me(dict):
        ...     deep = defaultdict(dict)
        ...
        ...     foo = _getitem_property('foo')
        ...     deep_thing = _getitem_property('deep.thing')


        >>> me = Me()
        >>> me.foo
        None

        >>> me.foo = 10
        >>> me.foo
        10
        >>> me['foo']
        10

        >>> me.deep_thing = 42
        >>> me.deep_thing
        42
        >>> me.deep
        defaultdict(<type 'dict'>, {'thing': 42})
    """

    def __init__(self, keypath, doc=None):
        path, _, self.key = keypath.rpartition('.')
        self.path = path.split('.') if path else None
        self.__doc__ = doc

    def _path(self, obj):
        return reduce(lambda d, k: d[k], [obj] + self.path) if self.path else obj

    def __get__(self, obj, type=None):
        if obj is None:
            return type
        return self._path(obj).get(self.key)

    def __set__(self, obj, value):
        self._path(obj)[self.key] = value