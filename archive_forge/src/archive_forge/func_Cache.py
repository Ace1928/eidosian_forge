from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import hashlib
import logging
import os
import threading
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files as file_utils
def Cache(func):
    """Caches the result of a function."""
    cached_results = {}

    @functools.wraps(func)
    def ReturnCachedOrCallFunc(*args):
        try:
            return cached_results[args]
        except KeyError:
            result = func(*args)
            cached_results[args] = result
            return result
    ReturnCachedOrCallFunc.__wrapped__ = func
    return ReturnCachedOrCallFunc