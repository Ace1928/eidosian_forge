import warnings
from logging import Logger, getLogger
from typing import Optional, Type, Union
from scrapy.exceptions import NotConfigured, ScrapyDeprecationWarning
from scrapy.http.request import Request
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.utils.misc import load_object
from scrapy.utils.python import global_object_name
from scrapy.utils.response import response_status_message
class RetryMiddleware(metaclass=BackwardsCompatibilityMetaclass):

    def __init__(self, settings):
        if not settings.getbool('RETRY_ENABLED'):
            raise NotConfigured
        self.max_retry_times = settings.getint('RETRY_TIMES')
        self.retry_http_codes = set((int(x) for x in settings.getlist('RETRY_HTTP_CODES')))
        self.priority_adjust = settings.getint('RETRY_PRIORITY_ADJUST')
        try:
            self.exceptions_to_retry = self.__getattribute__('EXCEPTIONS_TO_RETRY')
        except AttributeError:
            self.exceptions_to_retry = tuple((load_object(x) if isinstance(x, str) else x for x in settings.getlist('RETRY_EXCEPTIONS')))

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def process_response(self, request, response, spider):
        if request.meta.get('dont_retry', False):
            return response
        if response.status in self.retry_http_codes:
            reason = response_status_message(response.status)
            return self._retry(request, reason, spider) or response
        return response

    def process_exception(self, request, exception, spider):
        if isinstance(exception, self.exceptions_to_retry) and (not request.meta.get('dont_retry', False)):
            return self._retry(request, exception, spider)

    def _retry(self, request, reason, spider):
        max_retry_times = request.meta.get('max_retry_times', self.max_retry_times)
        priority_adjust = request.meta.get('priority_adjust', self.priority_adjust)
        return get_retry_request(request, reason=reason, spider=spider, max_retry_times=max_retry_times, priority_adjust=priority_adjust)
    __getattr__ = backwards_compatibility_getattr