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
def _does_token_match(request_csrf_token, csrf_secret):
    """
    Return whether the given CSRF token matches the given CSRF secret, after
    unmasking the token if necessary.

    This function assumes that the request_csrf_token argument has been
    validated to have the correct length (CSRF_SECRET_LENGTH or
    CSRF_TOKEN_LENGTH characters) and allowed characters, and that if it has
    length CSRF_TOKEN_LENGTH, it is a masked secret.
    """
    if len(request_csrf_token) == CSRF_TOKEN_LENGTH:
        request_csrf_token = _unmask_cipher_token(request_csrf_token)
    assert len(request_csrf_token) == CSRF_SECRET_LENGTH
    return constant_time_compare(request_csrf_token, csrf_secret)