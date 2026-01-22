from __future__ import unicode_literals
import datetime
import functools
from google.api_core import datetime_helpers
class ExponentialTimeout(object):
    """A decorator that adds an exponentially increasing timeout argument.

    DEPRECATED: the concept of incrementing timeout exponentially has been
    deprecated. Use ``TimeToDeadlineTimeout`` instead.

    This is useful if a function is called multiple times. Each time the
    function is called this decorator will calculate a new timeout parameter
    based on the the number of times the function has been called.

    For example

    .. code-block:: python

    Args:
        initial (float): The initial timeout to pass.
        maximum (float): The maximum timeout for any one call.
        multiplier (float): The multiplier applied to the timeout for each
            invocation.
        deadline (Optional[float]): The overall deadline across all
            invocations. This is used to prevent a very large calculated
            timeout from pushing the overall execution time over the deadline.
            This is especially useful in conjunction with
            :mod:`google.api_core.retry`. If ``None``, the timeouts will not
            be adjusted to accommodate an overall deadline.
    """

    def __init__(self, initial=_DEFAULT_INITIAL_TIMEOUT, maximum=_DEFAULT_MAXIMUM_TIMEOUT, multiplier=_DEFAULT_TIMEOUT_MULTIPLIER, deadline=_DEFAULT_DEADLINE):
        self._initial = initial
        self._maximum = maximum
        self._multiplier = multiplier
        self._deadline = deadline

    def with_deadline(self, deadline):
        """Return a copy of this timeout with the given deadline.

        Args:
            deadline (float): The overall deadline across all invocations.

        Returns:
            ExponentialTimeout: A new instance with the given deadline.
        """
        return ExponentialTimeout(initial=self._initial, maximum=self._maximum, multiplier=self._multiplier, deadline=deadline)

    def __call__(self, func):
        """Apply the timeout decorator.

        Args:
            func (Callable): The function to apply the timeout argument to.
                This function must accept a timeout keyword argument.

        Returns:
            Callable: The wrapped function.
        """
        timeouts = _exponential_timeout_generator(self._initial, self._maximum, self._multiplier, self._deadline)

        @functools.wraps(func)
        def func_with_timeout(*args, **kwargs):
            """Wrapped function that adds timeout."""
            kwargs['timeout'] = next(timeouts)
            return func(*args, **kwargs)
        return func_with_timeout

    def __str__(self):
        return '<ExponentialTimeout initial={:.1f}, maximum={:.1f}, multiplier={:.1f}, deadline={:.1f}>'.format(self._initial, self._maximum, self._multiplier, self._deadline)