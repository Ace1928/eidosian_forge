from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.cloud import CloudRetry
import random
from functools import wraps
import syslog
import time
def _exponential_backoff(retries=10, delay=2, backoff=2, max_delay=60):
    """ Customizable exponential backoff strategy.
    Args:
        retries (int): Maximum number of times to retry a request.
        delay (float): Initial (base) delay.
        backoff (float): base of the exponent to use for exponential
            backoff.
        max_delay (int): Optional. If provided each delay generated is capped
            at this amount. Defaults to 60 seconds.
    Returns:
        Callable that returns a generator. This generator yields durations in
        seconds to be used as delays for an exponential backoff strategy.
    Usage:
        >>> backoff = _exponential_backoff()
        >>> backoff
        <function backoff_backoff at 0x7f0d939facf8>
        >>> list(backoff())
        [2, 4, 8, 16, 32, 60, 60, 60, 60, 60]
    """

    def backoff_gen():
        for retry in range(0, retries):
            sleep = delay * backoff ** retry
            yield (sleep if max_delay is None else min(sleep, max_delay))
    return backoff_gen