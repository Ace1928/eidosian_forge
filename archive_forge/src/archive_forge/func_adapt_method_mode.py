import asyncio
import logging
import types
from asgiref.sync import async_to_sync, iscoroutinefunction, sync_to_async
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured, MiddlewareNotUsed
from django.core.signals import request_finished
from django.db import connections, transaction
from django.urls import get_resolver, set_urlconf
from django.utils.log import log_response
from django.utils.module_loading import import_string
from .exception import convert_exception_to_response
def adapt_method_mode(self, is_async, method, method_is_async=None, debug=False, name=None):
    """
        Adapt a method to be in the correct "mode":
        - If is_async is False:
          - Synchronous methods are left alone
          - Asynchronous methods are wrapped with async_to_sync
        - If is_async is True:
          - Synchronous methods are wrapped with sync_to_async()
          - Asynchronous methods are left alone
        """
    if method_is_async is None:
        method_is_async = iscoroutinefunction(method)
    if debug and (not name):
        name = name or 'method %s()' % method.__qualname__
    if is_async:
        if not method_is_async:
            if debug:
                logger.debug('Synchronous handler adapted for %s.', name)
            return sync_to_async(method, thread_sensitive=True)
    elif method_is_async:
        if debug:
            logger.debug('Asynchronous handler adapted for %s.', name)
        return async_to_sync(method)
    return method