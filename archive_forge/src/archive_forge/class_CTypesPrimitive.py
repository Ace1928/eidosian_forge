import ctypes, ctypes.util, operator, sys
from . import model
class CTypesPrimitive(CTypesGenericPrimitive):
    __slots__ = ['_value']
    _ctype = ctype
    _reftypename = '%s &' % name
    kind = kind1

    def __init__(self, value):
        self._value = value

    @staticmethod
    def _create_ctype_obj(init):
        if init is None:
            return ctype()
        return ctype(CTypesPrimitive._to_ctypes(init))
    if kind == 'int' or kind == 'byte':

        @classmethod
        def _cast_from(cls, source):
            source = _cast_source_to_int(source)
            source = ctype(source).value
            return cls(source)

        def __int__(self):
            return self._value
    if kind == 'bool':

        @classmethod
        def _cast_from(cls, source):
            if not isinstance(source, (int, long, float)):
                source = _cast_source_to_int(source)
            return cls(bool(source))

        def __int__(self):
            return int(self._value)
    if kind == 'char':

        @classmethod
        def _cast_from(cls, source):
            source = _cast_source_to_int(source)
            source = bytechr(source & 255)
            return cls(source)

        def __int__(self):
            return ord(self._value)
    if kind == 'float':

        @classmethod
        def _cast_from(cls, source):
            if isinstance(source, float):
                pass
            elif isinstance(source, CTypesGenericPrimitive):
                if hasattr(source, '__float__'):
                    source = float(source)
                else:
                    source = int(source)
            else:
                source = _cast_source_to_int(source)
            source = ctype(source).value
            return cls(source)

        def __int__(self):
            return int(self._value)

        def __float__(self):
            return self._value
    _cast_to_integer = __int__
    if kind == 'int' or kind == 'byte' or kind == 'bool':

        @staticmethod
        def _to_ctypes(x):
            if not isinstance(x, (int, long)):
                if isinstance(x, CTypesData):
                    x = int(x)
                else:
                    raise TypeError('integer expected, got %s' % type(x).__name__)
            if ctype(x).value != x:
                if not is_signed and x < 0:
                    raise OverflowError('%s: negative integer' % name)
                else:
                    raise OverflowError('%s: integer out of bounds' % name)
            return x
    if kind == 'char':

        @staticmethod
        def _to_ctypes(x):
            if isinstance(x, bytes) and len(x) == 1:
                return x
            if isinstance(x, CTypesPrimitive):
                return x._value
            raise TypeError('character expected, got %s' % type(x).__name__)

        def __nonzero__(self):
            return ord(self._value) != 0
    else:

        def __nonzero__(self):
            return self._value != 0
    __bool__ = __nonzero__
    if kind == 'float':

        @staticmethod
        def _to_ctypes(x):
            if not isinstance(x, (int, long, float, CTypesData)):
                raise TypeError('float expected, got %s' % type(x).__name__)
            return ctype(x).value

    @staticmethod
    def _from_ctypes(value):
        return getattr(value, 'value', value)

    @staticmethod
    def _initialize(blob, init):
        blob.value = CTypesPrimitive._to_ctypes(init)
    if kind == 'char':

        def _to_string(self, maxlen):
            return self._value
    if kind == 'byte':

        def _to_string(self, maxlen):
            return chr(self._value & 255)