import re
import math
from typing import Any, Optional, SupportsFloat, SupportsInt, Union, Type
from ..helpers import NUMERIC_INF_OR_NAN, INVALID_NUMERIC, collapse_white_spaces
from .atomic_types import AnyAtomicType
class Float10(float, AnyAtomicType):
    name = 'float'
    xsd_version = '1.0'
    pattern = re.compile('^(?:[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[Ee][+-]?[0-9]+)? |[+-]?INF|NaN)$')

    def __new__(cls, value: Union[str, SupportsFloat]) -> 'Float10':
        if isinstance(value, str):
            value = collapse_white_spaces(value)
            if value in NUMERIC_INF_OR_NAN or (cls.xsd_version != '1.0' and value == '+INF'):
                if value == 'NaN':
                    try:
                        return float_nan
                    except NameError:
                        pass
            elif value.lower() in INVALID_NUMERIC:
                raise ValueError('invalid value {!r} for xs:{}'.format(value, cls.name))
        elif math.isnan(value):
            try:
                return float_nan
            except NameError:
                pass
        _value = super().__new__(cls, value)
        if _value > 3.4028235e+38:
            return super().__new__(cls, 'INF')
        elif _value < -3.4028235e+38:
            return super().__new__(cls, '-INF')
        elif -1e-37 < _value < 1e-37:
            return super().__new__(cls, -0.0 if str(_value).startswith('-') else 0.0)
        return _value

    def __hash__(self) -> int:
        return super(Float10, self).__hash__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            if super(Float10, self).__eq__(other):
                return True
            return math.isclose(self, other, rel_tol=1e-07, abs_tol=0.0)
        return super(Float10, self).__eq__(other)

    def __ne__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            if super(Float10, self).__eq__(other):
                return False
            return not math.isclose(self, other, rel_tol=1e-07, abs_tol=0.0)
        return super(Float10, self).__ne__(other)

    def __add__(self, other: object) -> Union[float, 'Float10', 'Float']:
        if isinstance(other, (self.__class__, int)) and (not isinstance(other, bool)):
            return self.__class__(super(Float10, self).__add__(other))
        elif isinstance(other, float):
            return super(Float10, self).__add__(other)
        return NotImplemented

    def __radd__(self, other: object) -> Union[float, 'Float10', 'Float']:
        if isinstance(other, (self.__class__, int)) and (not isinstance(other, bool)):
            return self.__class__(super(Float10, self).__radd__(other))
        elif isinstance(other, float):
            return super(Float10, self).__radd__(other)
        return NotImplemented

    def __sub__(self, other: object) -> Union[float, 'Float10', 'Float']:
        if isinstance(other, (self.__class__, int)) and (not isinstance(other, bool)):
            return self.__class__(super(Float10, self).__sub__(other))
        elif isinstance(other, float):
            return super(Float10, self).__sub__(other)
        return NotImplemented

    def __rsub__(self, other: object) -> Union[float, 'Float10', 'Float']:
        if isinstance(other, (self.__class__, int)) and (not isinstance(other, bool)):
            return self.__class__(super(Float10, self).__rsub__(other))
        elif isinstance(other, float):
            return super(Float10, self).__rsub__(other)
        return NotImplemented

    def __mul__(self, other: object) -> Union[float, 'Float10', 'Float']:
        if isinstance(other, (self.__class__, int)) and (not isinstance(other, bool)):
            return self.__class__(super(Float10, self).__mul__(other))
        elif isinstance(other, float):
            return super(Float10, self).__mul__(other)
        return NotImplemented

    def __rmul__(self, other: object) -> Union[float, 'Float10', 'Float']:
        if isinstance(other, (self.__class__, int)) and (not isinstance(other, bool)):
            return self.__class__(super(Float10, self).__rmul__(other))
        elif isinstance(other, float):
            return super(Float10, self).__rmul__(other)
        return NotImplemented

    def __truediv__(self, other: object) -> Union[float, 'Float10', 'Float']:
        if isinstance(other, (self.__class__, int)) and (not isinstance(other, bool)):
            return self.__class__(super(Float10, self).__truediv__(other))
        elif isinstance(other, float):
            return super(Float10, self).__truediv__(other)
        return NotImplemented

    def __rtruediv__(self, other: object) -> Union[float, 'Float10', 'Float']:
        if isinstance(other, (self.__class__, int)) and (not isinstance(other, bool)):
            return self.__class__(super(Float10, self).__rtruediv__(other))
        elif isinstance(other, float):
            return super(Float10, self).__rtruediv__(other)
        return NotImplemented

    def __mod__(self, other: object) -> Union[float, 'Float10', 'Float']:
        if isinstance(other, (self.__class__, int)) and (not isinstance(other, bool)):
            return self.__class__(super(Float10, self).__mod__(other))
        elif isinstance(other, float):
            return super(Float10, self).__mod__(other)
        return NotImplemented

    def __rmod__(self, other: object) -> Union[float, 'Float10', 'Float']:
        if isinstance(other, (self.__class__, int)) and (not isinstance(other, bool)):
            return self.__class__(super(Float10, self).__rmod__(other))
        elif isinstance(other, float):
            return super(Float10, self).__rmod__(other)
        return NotImplemented

    def __abs__(self) -> Union['Float10', 'Float']:
        return self.__class__(super(Float10, self).__abs__())