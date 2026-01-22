import six
import sys
import time
import traceback
import random
import asyncio
import functools
def _retry_if_exception_of_type(retryable_types):

    def _retry_if_exception_these_types(exception):
        return isinstance(exception, retryable_types)
    return _retry_if_exception_these_types