import functools
import logging
import random
import threading
import time
from oslo_utils import excutils
from oslo_utils import importutils
from oslo_utils import reflection
from oslo_db import exception
from oslo_db import options
class wrap_db_retry(object):
    """Retry db.api methods, if db_error raised

    Retry decorated db.api methods. This decorator catches db_error and retries
    function in a loop until it succeeds, or until maximum retries count
    will be reached.

    Keyword arguments:

    :param retry_interval: seconds between transaction retries
    :type retry_interval: int or float

    :param max_retries: max number of retries before an error is raised
    :type max_retries: int

    :param inc_retry_interval: determine increase retry interval or not
    :type inc_retry_interval: bool

    :param max_retry_interval: max interval value between retries
    :type max_retry_interval: int or float

    :param exception_checker: checks if an exception should trigger a retry
    :type exception_checker: callable

    :param jitter: determine increase retry interval use jitter or not, jitter
           is always interpreted as True for a DBDeadlockError
    :type jitter: bool
    """

    def __init__(self, retry_interval=1, max_retries=20, inc_retry_interval=True, max_retry_interval=10, retry_on_disconnect=False, retry_on_deadlock=False, exception_checker=lambda exc: False, jitter=False):
        super(wrap_db_retry, self).__init__()
        self.jitter = jitter
        self.db_error = (exception.RetryRequest,)
        self.exception_checker = exception_checker
        if retry_on_disconnect:
            self.db_error += (exception.DBConnectionError,)
        if retry_on_deadlock:
            self.db_error += (exception.DBDeadlock,)
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.inc_retry_interval = inc_retry_interval
        self.max_retry_interval = max_retry_interval

    def __call__(self, f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            sleep_time = next_interval = self.retry_interval
            remaining = self.max_retries
            while True:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    with excutils.save_and_reraise_exception() as ectxt:
                        expected = self._is_exception_expected(e)
                        if remaining > 0:
                            ectxt.reraise = not expected
                        else:
                            if expected:
                                LOG.exception('DB exceeded retry limit.')
                            if isinstance(e, exception.RetryRequest):
                                ectxt.type_ = type(e.inner_exc)
                                ectxt.value = e.inner_exc
                    LOG.debug('Performing DB retry for function %s', reflection.get_callable_name(f))
                    time.sleep(sleep_time)
                    if self.inc_retry_interval:
                        if isinstance(e, exception.DBDeadlock):
                            jitter = True
                        else:
                            jitter = self.jitter
                        sleep_time, next_interval = self._get_inc_interval(next_interval, jitter)
                    remaining -= 1
        return wrapper

    def _is_exception_expected(self, exc):
        if isinstance(exc, self.db_error):
            if not isinstance(exc, exception.RetryRequest):
                LOG.debug('DB error: %s', exc)
            return True
        return self.exception_checker(exc)

    def _get_inc_interval(self, n, jitter):
        n = n * 2
        if jitter:
            sleep_time = random.uniform(0, n)
        else:
            sleep_time = n
        return (min(sleep_time, self.max_retry_interval), n)