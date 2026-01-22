import logging
import random
import sys
import time
import traceback
from google.cloud.ml.util import _exceptions
from six import reraise
class FuzzedExponentialIntervals(object):
    """Iterable for intervals that are exponentially spaced, with fuzzing.

  On iteration, yields retry interval lengths, in seconds. Every iteration over
  this iterable will yield differently fuzzed interval lengths, as long as fuzz
  is nonzero.

  Args:
    initial_delay_secs: The delay before the first retry, in seconds.
    num_retries: The total number of times to retry.
    factor: The exponential factor to use on subsequent retries.
      Default is 2 (doubling).
    fuzz: A value between 0 and 1, indicating the fraction of fuzz. For a
      given delay d, the fuzzed delay is randomly chosen between
      [(1 - fuzz) * d, d].
    max_delay_sec: Maximum delay (in seconds). After this limit is reached,
      further tries use max_delay_sec instead of exponentially increasing
      the time. Defaults to 5 minutes.
  """

    def __init__(self, initial_delay_secs, num_retries, factor=2, fuzz=0.5, max_delay_secs=30):
        self._initial_delay_secs = initial_delay_secs
        self._num_retries = num_retries
        self._factor = factor
        if not 0 <= fuzz <= 1:
            raise ValueError('Fuzz parameter expected to be in [0, 1] range.')
        self._fuzz = fuzz
        self._max_delay_secs = max_delay_secs

    def __iter__(self):
        current_delay_secs = min(self._max_delay_secs, self._initial_delay_secs)
        for _ in range(self._num_retries):
            fuzz_multiplier = 1 - self._fuzz + random.random() * self._fuzz
            yield (current_delay_secs * fuzz_multiplier)
            current_delay_secs = min(self._max_delay_secs, current_delay_secs * self._factor)