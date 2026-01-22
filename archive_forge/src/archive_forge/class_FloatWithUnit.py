from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
class FloatWithUnit(float):
    """Subclasses float to attach a unit type. Typically, you should use the
    pre-defined unit type subclasses such as Energy, Length, etc. instead of
    using FloatWithUnit directly.

    Supports conversion, addition and subtraction of the same unit type. E.g.,
    1 m + 20 cm will be automatically converted to 1.2 m (units follow the
    leftmost quantity). Note that FloatWithUnit does not override the eq
    method for float, i.e., units are not checked when testing for equality.
    The reason is to allow this class to be used transparently wherever floats
    are expected.

    >>> e = Energy(1.1, "Ha")
    >>> a = Energy(1.1, "Ha")
    >>> b = Energy(3, "eV")
    >>> c = a + b
    >>> print(c)
    1.2102479761938871 Ha
    >>> c.to("eV")
    32.932522246000005 eV
    """

    def __init__(self, val: float | Number, unit: str, unit_type: str | None=None) -> None:
        """Initializes a float with unit.

        Args:
            val (float): Value
            unit (Unit): A unit. E.g., "C".
            unit_type (str): A type of unit. E.g., "charge"
        """
        if unit_type is not None and str(unit) not in ALL_UNITS[unit_type]:
            raise UnitError(f'{unit} is not a supported unit for {unit_type}')
        self._unit = Unit(unit)
        self._unit_type = unit_type

    def __new__(cls, val, unit, unit_type=None) -> Self:
        """Overrides __new__ since we are subclassing a Python primitive."""
        new = float.__new__(cls, val)
        new._unit = Unit(unit)
        new._unit_type = unit_type
        return new

    def __str__(self) -> str:
        return f'{super().__str__()} {self._unit}'

    def __add__(self, other):
        if not hasattr(other, 'unit_type'):
            return super().__add__(other)
        if other.unit_type != self._unit_type:
            raise UnitError('Adding different types of units is not allowed')
        val = other
        if other.unit != self._unit:
            val = other.to(self._unit)
        return FloatWithUnit(float(self) + val, unit_type=self._unit_type, unit=self._unit)

    def __sub__(self, other):
        if not hasattr(other, 'unit_type'):
            return super().__sub__(other)
        if other.unit_type != self._unit_type:
            raise UnitError('Subtracting different units is not allowed')
        val = other
        if other.unit != self._unit:
            val = other.to(self._unit)
        return FloatWithUnit(float(self) - val, unit_type=self._unit_type, unit=self._unit)

    def __mul__(self, other):
        if not isinstance(other, FloatWithUnit):
            return FloatWithUnit(float(self) * other, unit_type=self._unit_type, unit=self._unit)
        return FloatWithUnit(float(self) * other, unit_type=None, unit=self._unit * other._unit)

    def __rmul__(self, other):
        if not isinstance(other, FloatWithUnit):
            return FloatWithUnit(float(self) * other, unit_type=self._unit_type, unit=self._unit)
        return FloatWithUnit(float(self) * other, unit_type=None, unit=self._unit * other._unit)

    def __pow__(self, i):
        return FloatWithUnit(float(self) ** i, unit_type=None, unit=self._unit ** i)

    def __truediv__(self, other):
        val = super().__truediv__(other)
        if not isinstance(other, FloatWithUnit):
            return FloatWithUnit(val, unit_type=self._unit_type, unit=self._unit)
        return FloatWithUnit(val, unit_type=None, unit=self._unit / other._unit)

    def __neg__(self):
        return FloatWithUnit(super().__neg__(), unit_type=self._unit_type, unit=self._unit)

    def __getnewargs__(self):
        """Function used by pickle to recreate object."""
        if hasattr(self, '_unit_type'):
            args = (float(self), self._unit, self._unit_type)
        else:
            args = (float(self), self._unit, None)
        return args

    def __getstate__(self):
        state = self.__dict__.copy()
        state['val'] = float(self)
        return state

    def __setstate__(self, state):
        self._unit = state['_unit']

    @property
    def unit_type(self) -> str | None:
        """The type of unit. Energy, Charge, etc."""
        return self._unit_type

    @property
    def unit(self) -> Unit:
        """The unit, e.g., "eV"."""
        return self._unit

    @classmethod
    def from_str(cls, s: str) -> Self:
        """Parse string to FloatWithUnit.
        Example: Memory.from_str("1. Mb").
        """
        s = s.strip()
        for _idx, char in enumerate(s):
            if char.isalpha() or char.isspace():
                break
        else:
            raise ValueError(f'Unit is missing in string {s}')
        num, unit = (float(s[:_idx]), s[_idx:])
        for unit_type, dct in BASE_UNITS.items():
            if unit in dct:
                return cls(num, unit, unit_type=unit_type)
        return cls(num, unit, unit_type=None)

    def to(self, new_unit):
        """Conversion to a new_unit. Right now, only supports 1 to 1 mapping of
        units of each type.

        Args:
            new_unit: New unit type.

        Returns:
            A FloatWithUnit object in the new units.

        Example usage:
        >>> e = Energy(1.1, "eV")
        >>> e = Energy(1.1, "Ha")
        >>> e.to("eV")
        29.932522246 eV
        """
        return FloatWithUnit(self * self.unit.get_conversion_factor(new_unit), unit_type=self._unit_type, unit=new_unit)

    @property
    def as_base_units(self):
        """Returns this FloatWithUnit in base SI units, including derived units.

        Returns:
            A FloatWithUnit object in base SI units
        """
        return self.to(self.unit.as_base_units[0])

    @property
    def supported_units(self):
        """Supported units for specific unit type."""
        return tuple(ALL_UNITS[self._unit_type])