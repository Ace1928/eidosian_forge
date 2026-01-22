import datetime
from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.http import HttpResponseNotAllowed
from django.middleware.http import ConditionalGetMiddleware
from django.utils import timezone
from django.utils.cache import get_conditional_response
from django.utils.decorators import decorator_from_middleware
from django.utils.http import http_date, quote_etag
from django.utils.log import log_response
def _post_process_request(request, response, res_etag, res_last_modified):
    if request.method in ('GET', 'HEAD'):
        if res_last_modified and (not response.has_header('Last-Modified')):
            response.headers['Last-Modified'] = http_date(res_last_modified)
        if res_etag:
            response.headers.setdefault('ETag', res_etag)