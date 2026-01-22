import atexit
import os
import platform
import random
import sys
import threading
import time
import uuid
from collections import deque
import sentry_sdk
from sentry_sdk._compat import PY33, PY311
from sentry_sdk._lru_cache import LRUCache
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _set_initial_sampling_decision(self, sampling_context):
    """
        Sets the profile's sampling decision according to the following
        precedence rules:

        1. If the transaction to be profiled is not sampled, that decision
        will be used, regardless of anything else.

        2. Use `profiles_sample_rate` to decide.
        """
    if not self.sampled:
        logger.debug('[Profiling] Discarding profile because transaction is discarded.')
        self.sampled = False
        return
    if self.scheduler is None:
        logger.debug('[Profiling] Discarding profile because profiler was not started.')
        self.sampled = False
        return
    hub = self.hub or sentry_sdk.Hub.current
    client = hub.client
    if client is None:
        self.sampled = False
        return
    options = client.options
    if callable(options.get('profiles_sampler')):
        sample_rate = options['profiles_sampler'](sampling_context)
    elif options['profiles_sample_rate'] is not None:
        sample_rate = options['profiles_sample_rate']
    else:
        sample_rate = options['_experiments'].get('profiles_sample_rate')
    if sample_rate is None:
        logger.debug('[Profiling] Discarding profile because profiling was not enabled.')
        self.sampled = False
        return
    if not is_valid_sample_rate(sample_rate, source='Profiling'):
        logger.warning('[Profiling] Discarding profile because of invalid sample rate.')
        self.sampled = False
        return
    self.sampled = random.random() < float(sample_rate)
    if self.sampled:
        logger.debug('[Profiling] Initializing profile')
    else:
        logger.debug("[Profiling] Discarding profile because it's not included in the random sample (sample rate = {sample_rate})".format(sample_rate=float(sample_rate)))