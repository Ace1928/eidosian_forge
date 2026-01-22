import logging
import string
from collections import defaultdict
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import DisallowedHost, ImproperlyConfigured
from django.http import HttpHeaders, UnreadablePostError
from django.urls import get_callable
from django.utils.cache import patch_vary_headers
from django.utils.crypto import constant_time_compare, get_random_string
from django.utils.deprecation import MiddlewareMixin
from django.utils.functional import cached_property
from django.utils.http import is_same_domain
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile
@cached_property
def allowed_origin_subdomains(self):
    """
        A mapping of allowed schemes to list of allowed netlocs, where all
        subdomains of the netloc are allowed.
        """
    allowed_origin_subdomains = defaultdict(list)
    for parsed in (urlparse(origin) for origin in settings.CSRF_TRUSTED_ORIGINS if '*' in origin):
        allowed_origin_subdomains[parsed.scheme].append(parsed.netloc.lstrip('*'))
    return allowed_origin_subdomains