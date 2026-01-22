import functools
import inspect
import wrapt
from debtcollector import _utils
class removed_property(object):
    """Property descriptor that deprecates a property.

    This works like the ``@property`` descriptor but can be used instead to
    provide the same functionality and also interact with the :mod:`warnings`
    module to warn when a property is accessed, set and/or deleted.

    :param message: string used as ending contents of the deprecate message
    :param version: version string (represents the version this deprecation
                    was created in)
    :param removal_version: version string (represents the version this
                            deprecation will be removed in); a string
                            of '?' will denote this will be removed in
                            some future unknown version
    :param stacklevel: stacklevel used in the :func:`warnings.warn` function
                       to locate where the users code is when reporting the
                       deprecation call (the default being 3)
    :param category: the :mod:`warnings` category to use, defaults to
                     :py:class:`DeprecationWarning` if not provided
    """
    _PROPERTY_GONE_TPLS = {'set': "Setting the '%s' property is deprecated", 'get': "Reading the '%s' property is deprecated", 'delete': "Deleting the '%s' property is deprecated"}

    def __init__(self, fget=None, fset=None, fdel=None, doc=None, stacklevel=3, category=DeprecationWarning, version=None, removal_version=None, message=None):
        self.fset = fset
        self.fget = fget
        self.fdel = fdel
        self.stacklevel = stacklevel
        self.category = category
        self.version = version
        self.removal_version = removal_version
        self.message = message
        if doc is None and inspect.isfunction(fget):
            doc = getattr(fget, '__doc__', None)
        self._message_cache = {}
        self.__doc__ = doc

    def _fetch_message_from_cache(self, kind):
        try:
            out_message = self._message_cache[kind]
        except KeyError:
            prefix_tpl = self._PROPERTY_GONE_TPLS[kind]
            prefix = prefix_tpl % _fetch_first_result(self.fget, self.fset, self.fdel, _get_qualified_name, value_not_found='???')
            out_message = _utils.generate_message(prefix, message=self.message, version=self.version, removal_version=self.removal_version)
            self._message_cache[kind] = out_message
        return out_message

    def __call__(self, fget, **kwargs):
        self.fget = fget
        self.message = kwargs.get('message', self.message)
        self.version = kwargs.get('version', self.version)
        self.removal_version = kwargs.get('removal_version', self.removal_version)
        self.stacklevel = kwargs.get('stacklevel', self.stacklevel)
        self.category = kwargs.get('category', self.category)
        self.__doc__ = kwargs.get('doc', getattr(fget, '__doc__', self.__doc__))
        self._message_cache.clear()
        return self

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        out_message = self._fetch_message_from_cache('delete')
        _utils.deprecation(out_message, stacklevel=self.stacklevel, category=self.category)
        self.fdel(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        out_message = self._fetch_message_from_cache('set')
        _utils.deprecation(out_message, stacklevel=self.stacklevel, category=self.category)
        self.fset(obj, value)

    def __get__(self, obj, value):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError('unreadable attribute')
        out_message = self._fetch_message_from_cache('get')
        _utils.deprecation(out_message, stacklevel=self.stacklevel, category=self.category)
        return self.fget(obj)

    def getter(self, fget):
        o = type(self)(fget, self.fset, self.fdel, self.__doc__)
        o.message = self.message
        o.version = self.version
        o.stacklevel = self.stacklevel
        o.removal_version = self.removal_version
        o.category = self.category
        return o

    def setter(self, fset):
        o = type(self)(self.fget, fset, self.fdel, self.__doc__)
        o.message = self.message
        o.version = self.version
        o.stacklevel = self.stacklevel
        o.removal_version = self.removal_version
        o.category = self.category
        return o

    def deleter(self, fdel):
        o = type(self)(self.fget, self.fset, fdel, self.__doc__)
        o.message = self.message
        o.version = self.version
        o.stacklevel = self.stacklevel
        o.removal_version = self.removal_version
        o.category = self.category
        return o