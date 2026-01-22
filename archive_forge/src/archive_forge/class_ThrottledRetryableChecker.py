import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
class ThrottledRetryableChecker(BaseRetryableChecker):
    _THROTTLED_ERROR_CODES = ['Throttling', 'ThrottlingException', 'ThrottledException', 'RequestThrottledException', 'TooManyRequestsException', 'ProvisionedThroughputExceededException', 'TransactionInProgressException', 'RequestLimitExceeded', 'BandwidthLimitExceeded', 'LimitExceededException', 'RequestThrottled', 'SlowDown', 'PriorRequestNotComplete', 'EC2ThrottledException']

    def __init__(self, throttled_error_codes=None):
        if throttled_error_codes is None:
            throttled_error_codes = self._THROTTLED_ERROR_CODES[:]
        self._throttled_error_codes = throttled_error_codes

    def is_retryable(self, context):
        return context.get_error_code() in self._throttled_error_codes