import abc
import random
import typing
from pip._vendor.tenacity import _utils
class wait_exponential_jitter(wait_base):
    """Wait strategy that applies exponential backoff and jitter.

    It allows for a customized initial wait, maximum wait and jitter.

    This implements the strategy described here:
    https://cloud.google.com/storage/docs/retry-strategy

    The wait time is min(initial * 2**n + random.uniform(0, jitter), maximum)
    where n is the retry count.
    """

    def __init__(self, initial: float=1, max: float=_utils.MAX_WAIT, exp_base: float=2, jitter: float=1) -> None:
        self.initial = initial
        self.max = max
        self.exp_base = exp_base
        self.jitter = jitter

    def __call__(self, retry_state: 'RetryCallState') -> float:
        jitter = random.uniform(0, self.jitter)
        try:
            exp = self.exp_base ** (retry_state.attempt_number - 1)
            result = self.initial * exp + jitter
        except OverflowError:
            result = self.max
        return max(0, min(result, self.max))