from abc import ABC, abstractmethod
from pennylane.operation import Operation
class ErrorOperation(Operation):
    """Base class that represents quantum operations which carry some form of algorithmic error.

    .. note::
        Child classes must implement the :func:`~.ErrorOperation.error` method which computes
        the error of the operation.
    """

    @property
    @abstractmethod
    def error(self) -> AlgorithmicError:
        """Computes the error of the operation.

        Returns:
            AlgorithmicError: The error.
        """