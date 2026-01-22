from typing import Type
import enum
import warnings
class EnumRule:
    """A marshal for converting between integer values and enum values."""

    def __init__(self, enum_class: Type[enum.IntEnum]):
        self._enum = enum_class

    def to_python(self, value, *, absent: bool=None):
        if isinstance(value, int) and (not isinstance(value, self._enum)):
            try:
                return self._enum(value)
            except ValueError:
                warnings.warn('Unrecognized {name} enum value: {value}'.format(name=self._enum.__name__, value=value))
        return value

    def to_proto(self, value):
        if isinstance(value, self._enum):
            return value.value
        if isinstance(value, str):
            return self._enum[value].value
        return value