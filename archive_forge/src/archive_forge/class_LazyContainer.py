from pprint import pformat
from .py3compat import MutableMapping
class LazyContainer(object):
    __slots__ = ['subcon', 'stream', 'pos', 'context', '_value']

    def __init__(self, subcon, stream, pos, context):
        self.subcon = subcon
        self.stream = stream
        self.pos = pos
        self.context = context
        self._value = NotImplemented

    def __eq__(self, other):
        try:
            return self._value == other._value
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.__pretty_str__()

    def __pretty_str__(self, nesting=1, indentation='    '):
        if self._value is NotImplemented:
            text = '<unread>'
        elif hasattr(self._value, '__pretty_str__'):
            text = self._value.__pretty_str__(nesting, indentation)
        else:
            text = str(self._value)
        return '%s: %s' % (self.__class__.__name__, text)

    def read(self):
        self.stream.seek(self.pos)
        return self.subcon._parse(self.stream, self.context)

    def dispose(self):
        self.subcon = None
        self.stream = None
        self.context = None
        self.pos = None

    def _get_value(self):
        if self._value is NotImplemented:
            self._value = self.read()
        return self._value
    value = property(_get_value)
    has_value = property(lambda self: self._value is not NotImplemented)