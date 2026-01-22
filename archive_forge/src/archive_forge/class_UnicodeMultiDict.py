import cgi
import copy
import sys
from collections.abc import MutableMapping as DictMixin
class UnicodeMultiDict(DictMixin):
    """
    A MultiDict wrapper that decodes returned values to unicode on the
    fly. Decoding is not applied to assigned values.

    The key/value contents are assumed to be ``str``/``strs`` or
    ``str``/``FieldStorages`` (as is returned by the ``paste.request.parse_``
    functions).

    Can optionally also decode keys when the ``decode_keys`` argument is
    True.

    ``FieldStorage`` instances are cloned, and the clone's ``filename``
    variable is decoded. Its ``name`` variable is decoded when ``decode_keys``
    is enabled.

    """

    def __init__(self, multi=None, encoding=None, errors='strict', decode_keys=False):
        self.multi = multi
        if encoding is None:
            encoding = sys.getdefaultencoding()
        self.encoding = encoding
        self.errors = errors
        self.decode_keys = decode_keys
        if self.decode_keys:
            items = self.multi._items
            for index, item in enumerate(items):
                key, value = item
                key = self._encode_key(key)
                items[index] = (key, value)

    def _encode_key(self, key):
        if self.decode_keys:
            try:
                key = key.encode(self.encoding, self.errors)
            except AttributeError:
                pass
        return key

    def _decode_key(self, key):
        if self.decode_keys:
            try:
                key = key.decode(self.encoding, self.errors)
            except AttributeError:
                pass
        return key

    def _decode_value(self, value):
        """
        Decode the specified value to unicode. Assumes value is a ``str`` or
        `FieldStorage`` object.

        ``FieldStorage`` objects are specially handled.
        """
        if isinstance(value, cgi.FieldStorage):
            decode_name = self.decode_keys and isinstance(value.name, bytes)
            if decode_name:
                value = copy.copy(value)
                if decode_name:
                    value.name = value.name.decode(self.encoding, self.errors)
        else:
            try:
                value = value.decode(self.encoding, self.errors)
            except AttributeError:
                pass
        return value

    def __getitem__(self, key):
        key = self._encode_key(key)
        return self._decode_value(self.multi.__getitem__(key))

    def __setitem__(self, key, value):
        key = self._encode_key(key)
        self.multi.__setitem__(key, value)

    def add(self, key, value):
        """
        Add the key and value, not overwriting any previous value.
        """
        key = self._encode_key(key)
        self.multi.add(key, value)

    def getall(self, key):
        """
        Return a list of all values matching the key (may be an empty list)
        """
        key = self._encode_key(key)
        return [self._decode_value(v) for v in self.multi.getall(key)]

    def getone(self, key):
        """
        Get one value matching the key, raising a KeyError if multiple
        values were found.
        """
        key = self._encode_key(key)
        return self._decode_value(self.multi.getone(key))

    def mixed(self):
        """
        Returns a dictionary where the values are either single
        values, or a list of values when a key/value appears more than
        once in this dictionary.  This is similar to the kind of
        dictionary often used to represent the variables in a web
        request.
        """
        unicode_mixed = {}
        for key, value in self.multi.mixed().items():
            if isinstance(value, list):
                value = [self._decode_value(value) for value in value]
            else:
                value = self._decode_value(value)
            unicode_mixed[self._decode_key(key)] = value
        return unicode_mixed

    def dict_of_lists(self):
        """
        Returns a dictionary where each key is associated with a
        list of values.
        """
        unicode_dict = {}
        for key, value in self.multi.dict_of_lists().items():
            value = [self._decode_value(value) for value in value]
            unicode_dict[self._decode_key(key)] = value
        return unicode_dict

    def __delitem__(self, key):
        key = self._encode_key(key)
        self.multi.__delitem__(key)

    def __contains__(self, key):
        key = self._encode_key(key)
        return self.multi.__contains__(key)
    has_key = __contains__

    def clear(self):
        self.multi.clear()

    def copy(self):
        return UnicodeMultiDict(self.multi.copy(), self.encoding, self.errors, decode_keys=self.decode_keys)

    def setdefault(self, key, default=None):
        key = self._encode_key(key)
        return self._decode_value(self.multi.setdefault(key, default))

    def pop(self, key, *args):
        key = self._encode_key(key)
        return self._decode_value(self.multi.pop(key, *args))

    def popitem(self):
        k, v = self.multi.popitem()
        return (self._decode_key(k), self._decode_value(v))

    def __repr__(self):
        items = ', '.join(['(%r, %r)' % v for v in self.items()])
        return '%s([%s])' % (self.__class__.__name__, items)

    def __len__(self):
        return self.multi.__len__()

    def keys(self):
        return [self._decode_key(k) for k in self.multi.iterkeys()]

    def iterkeys(self):
        for k in self.multi.iterkeys():
            yield self._decode_key(k)
    __iter__ = iterkeys

    def items(self):
        return [(self._decode_key(k), self._decode_value(v)) for k, v in self.multi.items()]

    def iteritems(self):
        for k, v in self.multi.items():
            yield (self._decode_key(k), self._decode_value(v))

    def values(self):
        return [self._decode_value(v) for v in self.multi.itervalues()]

    def itervalues(self):
        for v in self.multi.itervalues():
            yield self._decode_value(v)