from __future__ import annotations
import numbers
class JSNumber:
    """Utility class for exposing JavaScript Number constants."""
    MAX_SAFE_INTEGER = (1 << 53) - 1
    MIN_SAFE_INTEGER = -((1 << 53) - 1)
    MAX_VALUE = 1.7976931348623157e+308
    MIN_VALUE = 5e-324
    MIN_NEGATIVE_VALUE = -MAX_VALUE

    @classmethod
    def validate_int_bounds(cls, value: int, value_name: str | None=None) -> None:
        """Validate that an int value can be represented with perfect precision
        by a JavaScript Number.

        Parameters
        ----------
        value : int
        value_name : str or None
            The name of the value parameter. If specified, this will be used
            in any exception that is thrown.

        Raises
        ------
        JSNumberBoundsException
            Raised with a human-readable explanation if the value falls outside
            JavaScript int bounds.

        """
        if value_name is None:
            value_name = 'value'
        if value < cls.MIN_SAFE_INTEGER:
            raise JSNumberBoundsException(f'{value_name} ({value}) must be >= -((1 << 53) - 1)')
        elif value > cls.MAX_SAFE_INTEGER:
            raise JSNumberBoundsException(f'{value_name} ({value}) must be <= (1 << 53) - 1')

    @classmethod
    def validate_float_bounds(cls, value: int | float, value_name: str | None) -> None:
        """Validate that a float value can be represented by a JavaScript Number.

        Parameters
        ----------
        value : float
        value_name : str or None
            The name of the value parameter. If specified, this will be used
            in any exception that is thrown.

        Raises
        ------
        JSNumberBoundsException
            Raised with a human-readable explanation if the value falls outside
            JavaScript float bounds.

        """
        if value_name is None:
            value_name = 'value'
        if not isinstance(value, (numbers.Integral, float)):
            raise JSNumberBoundsException(f'{value_name} ({value}) is not a float')
        elif value < cls.MIN_NEGATIVE_VALUE:
            raise JSNumberBoundsException(f'{value_name} ({value}) must be >= -1.797e+308')
        elif value > cls.MAX_VALUE:
            raise JSNumberBoundsException(f'{value_name} ({value}) must be <= 1.797e+308')