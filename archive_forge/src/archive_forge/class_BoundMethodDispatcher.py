from __future__ import absolute_import, division, unicode_literals
from types import ModuleType
from six import text_type, PY3
class BoundMethodDispatcher(Mapping):
    """Wraps a MethodDispatcher, binding its return values to `instance`"""

    def __init__(self, instance, dispatcher):
        self.instance = instance
        self.dispatcher = dispatcher

    def __getitem__(self, key):
        return self.dispatcher[key].__get__(self.instance)

    def get(self, key, default):
        if key in self.dispatcher:
            return self[key]
        else:
            return default

    def __iter__(self):
        return iter(self.dispatcher)

    def __len__(self):
        return len(self.dispatcher)

    def __contains__(self, key):
        return key in self.dispatcher