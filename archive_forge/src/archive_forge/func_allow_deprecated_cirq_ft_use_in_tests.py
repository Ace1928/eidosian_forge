import functools
import unittest.mock
import os
from typing import Callable, Type
from cirq._compat import deprecated, deprecated_class
def allow_deprecated_cirq_ft_use_in_tests(func):
    """Decorator to allow using deprecated classes and functions in Tests and suppress warnings."""

    @functools.wraps(func)
    @unittest.mock.patch.dict(os.environ, ALLOW_DEPRECATION_IN_TEST='True')
    def wrapper(*args, **kwargs):
        from cirq.testing import assert_logs
        import logging
        with assert_logs(min_level=logging.WARNING, max_level=logging.WARNING, count=None) as logs:
            ret_val = func(*args, **kwargs)
        for log in logs:
            msg = log.getMessage()
            if _DEPRECATION_FIX_MSG in msg:
                assert _DEPRECATION_DEADLINE in msg
        return ret_val
    return wrapper