from __future__ import annotations
from typing import Callable
from cvxpy.expressions.constants.parameter import Parameter
class CallbackParam(Parameter):
    """
    A parameter whose value is derived from a callback function.

    Enables writing replacing expression that would not be DPP
    by a new parameter that automatically updates its value.

    Example:
    With p and q parameters, p * q is not DPP, but
    pq = CallbackParameter(callback=lambda: p.value * q.value) is DPP.

    This is useful when only p and q should be exposed
    to the user, but pq is needed internally.
    """

    def __init__(self, callback: Callable, shape: int | tuple[int, ...]=(), **kwargs) -> None:
        """
        callback: function that returns the value of the parameter.
        """
        self._callback = callback
        super(CallbackParam, self).__init__(shape, **kwargs)

    @property
    def value(self):
        """Evaluate the callback to get the value.
        """
        return self._validate_value(self._callback())

    @value.setter
    def value(self, _val):
        raise NotImplementedError('Cannot set the value of a CallbackParam.')