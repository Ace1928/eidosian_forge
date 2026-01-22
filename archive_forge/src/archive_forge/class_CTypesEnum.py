import ctypes, ctypes.util, operator, sys
from . import model
class CTypesEnum(CTypesInt):
    __slots__ = []
    _reftypename = '%s &' % name

    def _get_own_repr(self):
        value = self._value
        try:
            return '%d: %s' % (value, reverse_mapping[value])
        except KeyError:
            return str(value)

    def _to_string(self, maxlen):
        value = self._value
        try:
            return reverse_mapping[value]
        except KeyError:
            return str(value)