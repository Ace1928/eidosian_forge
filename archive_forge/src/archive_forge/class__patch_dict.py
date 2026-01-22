from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
class _patch_dict(object):
    """
    Patch a dictionary, or dictionary like object, and restore the dictionary
    to its original state after the test.

    `in_dict` can be a dictionary or a mapping like container. If it is a
    mapping then it must at least support getting, setting and deleting items
    plus iterating over keys.

    `in_dict` can also be a string specifying the name of the dictionary, which
    will then be fetched by importing it.

    `values` can be a dictionary of values to set in the dictionary. `values`
    can also be an iterable of `(key, value)` pairs.

    If `clear` is True then the dictionary will be cleared before the new
    values are set.

    `patch.dict` can also be called with arbitrary keyword arguments to set
    values in the dictionary::

        with patch.dict('sys.modules', mymodule=Mock(), other_module=Mock()):
            ...

    `patch.dict` can be used as a context manager, decorator or class
    decorator. When used as a class decorator `patch.dict` honours
    `patch.TEST_PREFIX` for choosing which methods to wrap.
    """

    def __init__(self, in_dict, values=(), clear=False, **kwargs):
        if isinstance(in_dict, basestring):
            in_dict = _importer(in_dict)
        self.in_dict = in_dict
        self.values = dict(values)
        self.values.update(kwargs)
        self.clear = clear
        self._original = None

    def __call__(self, f):
        if isinstance(f, ClassTypes):
            return self.decorate_class(f)

        @wraps(f)
        def _inner(*args, **kw):
            self._patch_dict()
            try:
                return f(*args, **kw)
            finally:
                self._unpatch_dict()
        return _inner

    def decorate_class(self, klass):
        for attr in dir(klass):
            attr_value = getattr(klass, attr)
            if attr.startswith(patch.TEST_PREFIX) and hasattr(attr_value, '__call__'):
                decorator = _patch_dict(self.in_dict, self.values, self.clear)
                decorated = decorator(attr_value)
                setattr(klass, attr, decorated)
        return klass

    def __enter__(self):
        """Patch the dict."""
        self._patch_dict()

    def _patch_dict(self):
        values = self.values
        in_dict = self.in_dict
        clear = self.clear
        try:
            original = in_dict.copy()
        except AttributeError:
            original = {}
            for key in in_dict:
                original[key] = in_dict[key]
        self._original = original
        if clear:
            _clear_dict(in_dict)
        try:
            in_dict.update(values)
        except AttributeError:
            for key in values:
                in_dict[key] = values[key]

    def _unpatch_dict(self):
        in_dict = self.in_dict
        original = self._original
        _clear_dict(in_dict)
        try:
            in_dict.update(original)
        except AttributeError:
            for key in original:
                in_dict[key] = original[key]

    def __exit__(self, *args):
        """Unpatch the dict."""
        self._unpatch_dict()
        return False
    start = __enter__
    stop = __exit__