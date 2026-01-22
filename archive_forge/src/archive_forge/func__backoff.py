from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.cloud import CloudRetry
import random
from functools import wraps
import syslog
import time
@classmethod
def _backoff(cls, backoff_strategy, catch_extra_error_codes=None):
    """ Retry calling the Cloud decorated function using the provided
        backoff strategy.
        Args:
            backoff_strategy (callable): Callable that returns a generator. The
            generator should yield sleep times for each retry of the decorated
            function.
        """

    def deco(f):

        @wraps(f)
        def retry_func(*args, **kwargs):
            for delay in backoff_strategy():
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if isinstance(e, cls.base_class):
                        response_code = cls.status_code_from_exception(e)
                        if cls.found(response_code, catch_extra_error_codes):
                            msg = '{0}: Retrying in {1} seconds...'.format(str(e), delay)
                            syslog.syslog(syslog.LOG_INFO, msg)
                            time.sleep(delay)
                        else:
                            raise e
                    else:
                        raise e
            return f(*args, **kwargs)
        return retry_func
    return deco