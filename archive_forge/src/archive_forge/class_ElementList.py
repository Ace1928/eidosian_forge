from decimal import Decimal
from boto.compat import filter, map
class ElementList(SimpleList):

    def start(self, *args, **kw):
        value = self._hint(parent=self._parent, **kw)
        self._value.append(value)
        return value

    def end(self, *args, **kw):
        pass