import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
class StandardRetryConditions(BaseRetryableChecker):
    """Concrete class that implements the standard retry policy checks.

    Specifically:

        not max_attempts and (transient or throttled or modeled_retry)

    """

    def __init__(self, max_attempts=DEFAULT_MAX_ATTEMPTS):
        self._max_attempts_checker = MaxAttemptsChecker(max_attempts)
        self._additional_checkers = OrRetryChecker([TransientRetryableChecker(), ThrottledRetryableChecker(), ModeledRetryableChecker(), OrRetryChecker([special.RetryIDPCommunicationError(), special.RetryDDBChecksumError()])])

    def is_retryable(self, context):
        return self._max_attempts_checker.is_retryable(context) and self._additional_checkers.is_retryable(context)