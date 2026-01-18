from decimal import Decimal
from functools import total_ordering
@classmethod
def unit_attname(cls, unit_str):
    """
        Retrieve the unit attribute name for the given unit string.
        For example, if the given unit string is 'metre', return 'm'.
        Raise an AttributeError if an attribute cannot be found.
        """
    lower = unit_str.lower()
    if unit_str in cls.UNITS:
        return unit_str
    elif lower in cls.UNITS:
        return lower
    elif lower in cls.LALIAS:
        return cls.LALIAS[lower]
    else:
        raise AttributeError(f'Unknown unit type: {unit_str}')