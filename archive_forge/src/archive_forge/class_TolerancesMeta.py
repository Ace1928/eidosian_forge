from abc import ABCMeta
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT
class TolerancesMeta(ABCMeta):
    """Metaclass to handle tolerances"""

    def __init__(cls, *args, **kwargs):
        cls._ATOL_DEFAULT = ATOL_DEFAULT
        cls._RTOL_DEFAULT = RTOL_DEFAULT
        cls._MAX_TOL = 0.0001
        super().__init__(cls, args, kwargs)

    def _check_value(cls, value, value_name):
        """Check if value is within valid ranges"""
        if value < 0:
            raise QiskitError(f'Invalid {value_name} ({value}) must be non-negative.')
        if value > cls._MAX_TOL:
            raise QiskitError(f'Invalid {value_name} ({value}) must be less than {cls._MAX_TOL}.')

    @property
    def atol(cls):
        """Default absolute tolerance parameter for float comparisons."""
        return cls._ATOL_DEFAULT

    @atol.setter
    def atol(cls, value):
        """Set default absolute tolerance parameter for float comparisons."""
        cls._check_value(value, 'atol')
        cls._ATOL_DEFAULT = value

    @property
    def rtol(cls):
        """Default relative tolerance parameter for float comparisons."""
        return cls._RTOL_DEFAULT

    @rtol.setter
    def rtol(cls, value):
        """Set default relative tolerance parameter for float comparisons."""
        cls._check_value(value, 'rtol')
        cls._RTOL_DEFAULT = value