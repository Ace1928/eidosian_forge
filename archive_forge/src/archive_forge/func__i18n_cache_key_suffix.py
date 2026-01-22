import time
from collections import defaultdict
from hashlib import md5
from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse, HttpResponseNotModified
from django.utils.http import http_date, parse_etags, parse_http_date_safe, quote_etag
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import get_current_timezone_name
from django.utils.translation import get_language
def _i18n_cache_key_suffix(request, cache_key):
    """If necessary, add the current locale or time zone to the cache key."""
    if settings.USE_I18N:
        cache_key += '.%s' % getattr(request, 'LANGUAGE_CODE', get_language())
    if settings.USE_TZ:
        cache_key += '.%s' % get_current_timezone_name()
    return cache_key