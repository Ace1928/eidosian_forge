from __future__ import annotations
from .mixins import ImmutableDictMixin
from .mixins import UpdateDictMixin
from .. import http
class _CacheControl(UpdateDictMixin, dict):
    """Subclass of a dict that stores values for a Cache-Control header.  It
    has accessors for all the cache-control directives specified in RFC 2616.
    The class does not differentiate between request and response directives.

    Because the cache-control directives in the HTTP header use dashes the
    python descriptors use underscores for that.

    To get a header of the :class:`CacheControl` object again you can convert
    the object into a string or call the :meth:`to_header` method.  If you plan
    to subclass it and add your own items have a look at the sourcecode for
    that class.

    .. versionchanged:: 2.1.0
        Setting int properties such as ``max_age`` will convert the
        value to an int.

    .. versionchanged:: 0.4

       Setting `no_cache` or `private` to boolean `True` will set the implicit
       none-value which is ``*``:

       >>> cc = ResponseCacheControl()
       >>> cc.no_cache = True
       >>> cc
       <ResponseCacheControl 'no-cache'>
       >>> cc.no_cache
       '*'
       >>> cc.no_cache = None
       >>> cc
       <ResponseCacheControl ''>

       In versions before 0.5 the behavior documented here affected the now
       no longer existing `CacheControl` class.
    """
    no_cache = cache_control_property('no-cache', '*', None)
    no_store = cache_control_property('no-store', None, bool)
    max_age = cache_control_property('max-age', -1, int)
    no_transform = cache_control_property('no-transform', None, None)

    def __init__(self, values=(), on_update=None):
        dict.__init__(self, values or ())
        self.on_update = on_update
        self.provided = values is not None

    def _get_cache_value(self, key, empty, type):
        """Used internally by the accessor properties."""
        if type is bool:
            return key in self
        if key in self:
            value = self[key]
            if value is None:
                return empty
            elif type is not None:
                try:
                    value = type(value)
                except ValueError:
                    pass
            return value
        return None

    def _set_cache_value(self, key, value, type):
        """Used internally by the accessor properties."""
        if type is bool:
            if value:
                self[key] = None
            else:
                self.pop(key, None)
        elif value is None:
            self.pop(key, None)
        elif value is True:
            self[key] = None
        elif type is not None:
            self[key] = type(value)
        else:
            self[key] = value

    def _del_cache_value(self, key):
        """Used internally by the accessor properties."""
        if key in self:
            del self[key]

    def to_header(self):
        """Convert the stored values into a cache control header."""
        return http.dump_header(self)

    def __str__(self):
        return self.to_header()

    def __repr__(self):
        kv_str = ' '.join((f'{k}={v!r}' for k, v in sorted(self.items())))
        return f'<{type(self).__name__} {kv_str}>'
    cache_property = staticmethod(cache_control_property)