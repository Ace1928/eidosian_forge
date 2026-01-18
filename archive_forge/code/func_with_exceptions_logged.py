from abc import ABC
import collections
import enum
import functools
import logging
def with_exceptions_logged(behavior, message):
    """Wraps a callable in a try-except that logs any exceptions it raises.

    Args:
      behavior: Any callable.
      message: A string to log if the behavior raises an exception.

    Returns:
      A callable that when executed invokes the given behavior. The returned
        callable takes the same arguments as the given behavior but returns a
        future.Outcome describing whether the given behavior returned a value or
        raised an exception.
    """

    @functools.wraps(behavior)
    def wrapped_behavior(*args, **kwargs):
        return _call_logging_exceptions(behavior, message, *args, **kwargs)
    return wrapped_behavior