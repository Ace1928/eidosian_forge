from __future__ import absolute_import
import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile
from ..exceptions import (
from ..packages import six
class _RetryMeta(type):

    @property
    def DEFAULT_METHOD_WHITELIST(cls):
        warnings.warn("Using 'Retry.DEFAULT_METHOD_WHITELIST' is deprecated and will be removed in v2.0. Use 'Retry.DEFAULT_ALLOWED_METHODS' instead", DeprecationWarning)
        return cls.DEFAULT_ALLOWED_METHODS

    @DEFAULT_METHOD_WHITELIST.setter
    def DEFAULT_METHOD_WHITELIST(cls, value):
        warnings.warn("Using 'Retry.DEFAULT_METHOD_WHITELIST' is deprecated and will be removed in v2.0. Use 'Retry.DEFAULT_ALLOWED_METHODS' instead", DeprecationWarning)
        cls.DEFAULT_ALLOWED_METHODS = value

    @property
    def DEFAULT_REDIRECT_HEADERS_BLACKLIST(cls):
        warnings.warn("Using 'Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST' is deprecated and will be removed in v2.0. Use 'Retry.DEFAULT_REMOVE_HEADERS_ON_REDIRECT' instead", DeprecationWarning)
        return cls.DEFAULT_REMOVE_HEADERS_ON_REDIRECT

    @DEFAULT_REDIRECT_HEADERS_BLACKLIST.setter
    def DEFAULT_REDIRECT_HEADERS_BLACKLIST(cls, value):
        warnings.warn("Using 'Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST' is deprecated and will be removed in v2.0. Use 'Retry.DEFAULT_REMOVE_HEADERS_ON_REDIRECT' instead", DeprecationWarning)
        cls.DEFAULT_REMOVE_HEADERS_ON_REDIRECT = value

    @property
    def BACKOFF_MAX(cls):
        warnings.warn("Using 'Retry.BACKOFF_MAX' is deprecated and will be removed in v2.0. Use 'Retry.DEFAULT_BACKOFF_MAX' instead", DeprecationWarning)
        return cls.DEFAULT_BACKOFF_MAX

    @BACKOFF_MAX.setter
    def BACKOFF_MAX(cls, value):
        warnings.warn("Using 'Retry.BACKOFF_MAX' is deprecated and will be removed in v2.0. Use 'Retry.DEFAULT_BACKOFF_MAX' instead", DeprecationWarning)
        cls.DEFAULT_BACKOFF_MAX = value