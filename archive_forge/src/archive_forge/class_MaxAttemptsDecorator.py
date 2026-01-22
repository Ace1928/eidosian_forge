import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
class MaxAttemptsDecorator(BaseChecker):
    """Allow retries up to a maximum number of attempts.

    This will pass through calls to the decorated retry checker, provided
    that the number of attempts does not exceed max_attempts.  It will
    also catch any retryable_exceptions passed in.  Once max_attempts has
    been exceeded, then False will be returned or the retryable_exceptions
    that was previously being caught will be raised.

    """

    def __init__(self, checker, max_attempts, retryable_exceptions=None):
        self._checker = checker
        self._max_attempts = max_attempts
        self._retryable_exceptions = retryable_exceptions

    def __call__(self, attempt_number, response, caught_exception, retries_context):
        if retries_context:
            retries_context['max'] = max(retries_context.get('max', 0), self._max_attempts)
        should_retry = self._should_retry(attempt_number, response, caught_exception)
        if should_retry:
            if attempt_number >= self._max_attempts:
                if response is not None and 'ResponseMetadata' in response[1]:
                    response[1]['ResponseMetadata']['MaxAttemptsReached'] = True
                logger.debug('Reached the maximum number of retry attempts: %s', attempt_number)
                return False
            else:
                return should_retry
        else:
            return False

    def _should_retry(self, attempt_number, response, caught_exception):
        if self._retryable_exceptions and attempt_number < self._max_attempts:
            try:
                return self._checker(attempt_number, response, caught_exception)
            except self._retryable_exceptions as e:
                logger.debug('retry needed, retryable exception caught: %s', e, exc_info=True)
                return True
        else:
            return self._checker(attempt_number, response, caught_exception)