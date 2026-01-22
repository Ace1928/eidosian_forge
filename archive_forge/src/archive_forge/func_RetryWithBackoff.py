from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import posixpath
import sys
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import client_deployinfo
import six
from six.moves import urllib
def RetryWithBackoff(func, retry_notify_func, initial_delay=1, backoff_factor=2, max_delay=60, max_tries=20, raise_on_timeout=True):
    """Calls a function multiple times, backing off more and more each time.

  Args:
    func: f() -> (bool, value), A function that performs some operation that
      should be retried a number of times upon failure. If the first tuple
      element is True, we'll immediately return (True, value). If False, we'll
      delay a bit and try again, unless we've hit the 'max_tries' limit, in
      which case we'll return (False, value).
    retry_notify_func: f(value, delay) -> None, This function will be called
      immediately before the next retry delay.  'value' is the value returned
      by the last call to 'func'.  'delay' is the retry delay, in seconds
    initial_delay: int, Initial delay after first try, in seconds.
    backoff_factor: int, Delay will be multiplied by this factor after each
      try.
    max_delay: int, Maximum delay, in seconds.
    max_tries: int, Maximum number of tries (the first one counts).
    raise_on_timeout: bool, True to raise an exception if the operation times
      out instead of returning False.

  Returns:
    What the last call to 'func' returned, which is of the form (done, value).
    If 'done' is True, you know 'func' returned True before we ran out of
    retries.  If 'done' is False, you know 'func' kept returning False and we
    ran out of retries.

  Raises:
    TimeoutError: If raise_on_timeout is True and max_tries is exhausted.
  """
    delay = initial_delay
    try_count = max_tries
    value = None
    while True:
        try_count -= 1
        done, value = func()
        if done:
            return (True, value)
        if try_count <= 0:
            if raise_on_timeout:
                raise TimeoutError()
            return (False, value)
        retry_notify_func(value, delay)
        time.sleep(delay)
        delay = min(delay * backoff_factor, max_delay)