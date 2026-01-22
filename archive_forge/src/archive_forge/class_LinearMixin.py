import sys
from abc import ABC, abstractmethod
from .multiply import MultiplyMixin
class LinearMixin(MultiplyMixin, ABC):
    """Abstract Mixin for linear operator.

    This class defines the following operator overloads:

        - ``+`` / ``__add__``
        - ``-`` / ``__sub__``
        - ``*`` / ``__rmul__`
        - ``/`` / ``__truediv__``
        - ``__neg__``

    The following abstract methods must be implemented by subclasses
    using this mixin

        - ``_add(self, other, qargs=None)``
        - ``_multiply(self, other)``
    """

    def __add__(self, other) -> Self:
        if not isinstance(other, type(self)) and other == 0:
            return self
        qargs = getattr(other, 'qargs', None)
        return self._add(other, qargs=qargs)

    def __radd__(self, other) -> Self:
        if not isinstance(other, type(self)) and other == 0:
            return self
        qargs = getattr(other, 'qargs', None)
        return self._add(other, qargs=qargs)

    def __sub__(self, other) -> Self:
        qargs = getattr(other, 'qargs', None)
        return self._add(-other, qargs=qargs)

    def __rsub__(self, other) -> Self:
        qargs = getattr(other, 'qargs', None)
        return (-self)._add(other, qargs=qargs)

    @abstractmethod
    def _add(self, other, qargs=None):
        """Return the CLASS self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (CLASS): an operator object.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            CLASS: the CLASS self + other.
        """