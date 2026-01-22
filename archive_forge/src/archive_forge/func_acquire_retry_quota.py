import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
def acquire_retry_quota(self, context):
    if self._is_timeout_error(context):
        capacity_amount = self._TIMEOUT_RETRY_REQUEST
    else:
        capacity_amount = self._RETRY_COST
    success = self._quota.acquire(capacity_amount)
    if success:
        context.request_context['retry_quota_capacity'] = capacity_amount
        return True
    context.add_retry_metadata(RetryQuotaReached=True)
    return False