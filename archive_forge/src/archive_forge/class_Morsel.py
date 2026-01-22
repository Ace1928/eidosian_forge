import re
import string
import types
class Morsel(dict):
    """A class to hold ONE (key, value) pair.

    In a cookie, each such pair may have several attributes, so this class is
    used to keep the attributes associated with the appropriate key,value pair.
    This class also includes a coded_value attribute, which is used to hold
    the network representation of the value.
    """
    _reserved = {'expires': 'expires', 'path': 'Path', 'comment': 'Comment', 'domain': 'Domain', 'max-age': 'Max-Age', 'secure': 'Secure', 'httponly': 'HttpOnly', 'version': 'Version', 'samesite': 'SameSite'}
    _flags = {'secure', 'httponly'}

    def __init__(self):
        self._key = self._value = self._coded_value = None
        for key in self._reserved:
            dict.__setitem__(self, key, '')

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    @property
    def coded_value(self):
        return self._coded_value

    def __setitem__(self, K, V):
        K = K.lower()
        if not K in self._reserved:
            raise CookieError('Invalid attribute %r' % (K,))
        dict.__setitem__(self, K, V)

    def setdefault(self, key, val=None):
        key = key.lower()
        if key not in self._reserved:
            raise CookieError('Invalid attribute %r' % (key,))
        return dict.setdefault(self, key, val)

    def __eq__(self, morsel):
        if not isinstance(morsel, Morsel):
            return NotImplemented
        return dict.__eq__(self, morsel) and self._value == morsel._value and (self._key == morsel._key) and (self._coded_value == morsel._coded_value)
    __ne__ = object.__ne__

    def copy(self):
        morsel = Morsel()
        dict.update(morsel, self)
        morsel.__dict__.update(self.__dict__)
        return morsel

    def update(self, values):
        data = {}
        for key, val in dict(values).items():
            key = key.lower()
            if key not in self._reserved:
                raise CookieError('Invalid attribute %r' % (key,))
            data[key] = val
        dict.update(self, data)

    def isReservedKey(self, K):
        return K.lower() in self._reserved

    def set(self, key, val, coded_val):
        if key.lower() in self._reserved:
            raise CookieError('Attempt to set a reserved key %r' % (key,))
        if not _is_legal_key(key):
            raise CookieError('Illegal key %r' % (key,))
        self._key = key
        self._value = val
        self._coded_value = coded_val

    def __getstate__(self):
        return {'key': self._key, 'value': self._value, 'coded_value': self._coded_value}

    def __setstate__(self, state):
        self._key = state['key']
        self._value = state['value']
        self._coded_value = state['coded_value']

    def output(self, attrs=None, header='Set-Cookie:'):
        return '%s %s' % (header, self.OutputString(attrs))
    __str__ = output

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.OutputString())

    def js_output(self, attrs=None):
        return '\n        <script type="text/javascript">\n        <!-- begin hiding\n        document.cookie = "%s";\n        // end hiding -->\n        </script>\n        ' % self.OutputString(attrs).replace('"', '\\"')

    def OutputString(self, attrs=None):
        result = []
        append = result.append
        append('%s=%s' % (self.key, self.coded_value))
        if attrs is None:
            attrs = self._reserved
        items = sorted(self.items())
        for key, value in items:
            if value == '':
                continue
            if key not in attrs:
                continue
            if key == 'expires' and isinstance(value, int):
                append('%s=%s' % (self._reserved[key], _getdate(value)))
            elif key == 'max-age' and isinstance(value, int):
                append('%s=%d' % (self._reserved[key], value))
            elif key == 'comment' and isinstance(value, str):
                append('%s=%s' % (self._reserved[key], _quote(value)))
            elif key in self._flags:
                if value:
                    append(str(self._reserved[key]))
            else:
                append('%s=%s' % (self._reserved[key], value))
        return _semispacejoin(result)
    __class_getitem__ = classmethod(types.GenericAlias)