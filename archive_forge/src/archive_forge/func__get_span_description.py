import functools
from typing import TYPE_CHECKING
from django import VERSION as DJANGO_VERSION
from django.core.cache import CacheHandler
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
def _get_span_description(method_name, args, kwargs):
    description = '{} '.format(method_name)
    if args is not None and len(args) >= 1:
        description += text_type(args[0])
    elif kwargs is not None and 'key' in kwargs:
        description += text_type(kwargs['key'])
    return description