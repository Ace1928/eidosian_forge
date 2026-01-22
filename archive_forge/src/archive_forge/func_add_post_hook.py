import re
import sys
from functools import wraps
from inspect import getmembers
from types import CoroutineType
from typing import AsyncGenerator, Dict
from unittest import TestCase
from scrapy.http import Request
from scrapy.utils.python import get_spec
from scrapy.utils.spider import iterate_spider_output
def add_post_hook(self, request, results):
    if hasattr(self, 'post_process'):
        cb = request.callback

        @wraps(cb)
        def wrapper(response, **cb_kwargs):
            cb_result = cb(response, **cb_kwargs)
            if isinstance(cb_result, (AsyncGenerator, CoroutineType)):
                raise TypeError("Contracts don't support async callbacks")
            output = list(iterate_spider_output(cb_result))
            try:
                results.startTest(self.testcase_post)
                self.post_process(output)
                results.stopTest(self.testcase_post)
            except AssertionError:
                results.addFailure(self.testcase_post, sys.exc_info())
            except Exception:
                results.addError(self.testcase_post, sys.exc_info())
            else:
                results.addSuccess(self.testcase_post)
            finally:
                return output
        request.callback = wrapper
    return request