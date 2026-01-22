from decimal import Decimal
from boto.compat import filter, map
class DeclarativeType(object):

    def __init__(self, _hint=None, **kw):
        self._value = None
        if _hint is not None:
            self._hint = _hint
            return

        class JITResponse(ResponseElement):
            pass
        self._hint = JITResponse
        self._hint.__name__ = 'JIT_{0}/{1}'.format(self.__class__.__name__, hex(id(self._hint))[2:])
        for name, value in kw.items():
            setattr(self._hint, name, value)

    def __repr__(self):
        parent = getattr(self, '_parent', None)
        return '<{0}_{1}/{2}_{3}>'.format(self.__class__.__name__, parent and parent._name or '?', getattr(self, '_name', '?'), hex(id(self.__class__)))

    def setup(self, parent, name, *args, **kw):
        self._parent = parent
        self._name = name
        self._clone = self.__class__(_hint=self._hint)
        self._clone._parent = parent
        self._clone._name = name
        setattr(self._parent, self._name, self._clone)

    def start(self, *args, **kw):
        raise NotImplementedError

    def end(self, *args, **kw):
        raise NotImplementedError

    def teardown(self, *args, **kw):
        setattr(self._parent, self._name, self._value)