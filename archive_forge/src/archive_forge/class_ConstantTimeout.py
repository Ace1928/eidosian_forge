from __future__ import unicode_literals
import datetime
import functools
from google.api_core import datetime_helpers
class ConstantTimeout(object):
    """A decorator that adds a constant timeout argument.

    DEPRECATED: use ``TimeToDeadlineTimeout`` instead.

    This is effectively equivalent to
    ``functools.partial(func, timeout=timeout)``.

    Args:
        timeout (Optional[float]): the timeout (in seconds) to applied to the
            wrapped function. If `None`, the target function is expected to
            never timeout.
    """

    def __init__(self, timeout=None):
        self._timeout = timeout

    def __call__(self, func):
        """Apply the timeout decorator.

        Args:
            func (Callable): The function to apply the timeout argument to.
                This function must accept a timeout keyword argument.

        Returns:
            Callable: The wrapped function.
        """

        @functools.wraps(func)
        def func_with_timeout(*args, **kwargs):
            """Wrapped function that adds timeout."""
            kwargs['timeout'] = self._timeout
            return func(*args, **kwargs)
        return func_with_timeout

    def __str__(self):
        return '<ConstantTimeout timeout={:.1f}>'.format(self._timeout)