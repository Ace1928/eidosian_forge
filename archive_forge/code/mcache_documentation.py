import logging
import memcache
from saml2 import time_util
from saml2.cache import CacheError
from saml2.cache import TooOld
Return identifiers for all the subjects that are in the cache.

        :return: list of subject identifiers
        