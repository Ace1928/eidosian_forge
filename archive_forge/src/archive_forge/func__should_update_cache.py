from django.conf import settings
from django.core.cache import DEFAULT_CACHE_ALIAS, caches
from django.utils.cache import (
from django.utils.deprecation import MiddlewareMixin
def _should_update_cache(self, request, response):
    return hasattr(request, '_cache_update_cache') and request._cache_update_cache