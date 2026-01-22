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
def _generate_cache_header_key(key_prefix, request):
    """Return a cache key for the header cache."""
    url = md5(request.build_absolute_uri().encode('ascii'), usedforsecurity=False)
    cache_key = 'views.decorators.cache.cache_header.%s.%s' % (key_prefix, url.hexdigest())
    return _i18n_cache_key_suffix(request, cache_key)