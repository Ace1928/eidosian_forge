import binascii
import warnings
from webob.compat import (
class GetDict(MultiDict):

    def __init__(self, data, env):
        self.env = env
        MultiDict.__init__(self, data)

    def on_change(self):
        e = lambda t: t.encode('utf8')
        data = [(e(k), e(v)) for k, v in self.items()]
        qs = url_encode(data)
        self.env['QUERY_STRING'] = qs
        self.env['webob._parsed_query_vars'] = (self, qs)

    def __setitem__(self, key, value):
        MultiDict.__setitem__(self, key, value)
        self.on_change()

    def add(self, key, value):
        MultiDict.add(self, key, value)
        self.on_change()

    def __delitem__(self, key):
        MultiDict.__delitem__(self, key)
        self.on_change()

    def clear(self):
        MultiDict.clear(self)
        self.on_change()

    def setdefault(self, key, default=None):
        result = MultiDict.setdefault(self, key, default)
        self.on_change()
        return result

    def pop(self, key, *args):
        result = MultiDict.pop(self, key, *args)
        self.on_change()
        return result

    def popitem(self):
        result = MultiDict.popitem(self)
        self.on_change()
        return result

    def update(self, *args, **kwargs):
        MultiDict.update(self, *args, **kwargs)
        self.on_change()

    def extend(self, *args, **kwargs):
        MultiDict.extend(self, *args, **kwargs)
        self.on_change()

    def __repr__(self):
        items = map('(%r, %r)'.__mod__, _hide_passwd(self.items()))
        return 'GET([%s])' % ', '.join(items)

    def copy(self):
        return MultiDict(self)