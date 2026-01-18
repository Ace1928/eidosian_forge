import contextlib
import hashlib
import os
from oslo_utils import reflection
from oslo_utils import strutils
from troveclient.compat import exceptions
from troveclient.compat import utils
def write_to_completion_cache(self, cache_type, val):
    cache = getattr(self, '_%s_cache' % cache_type, None)
    if cache:
        cache.write('%s\n' % val)