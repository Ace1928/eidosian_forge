import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
class OrRetryChecker(BaseRetryableChecker):

    def __init__(self, checkers):
        self._checkers = checkers

    def is_retryable(self, context):
        return any((checker.is_retryable(context) for checker in self._checkers))