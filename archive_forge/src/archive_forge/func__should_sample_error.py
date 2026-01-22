from importlib import import_module
import os
import uuid
import random
import socket
from sentry_sdk._compat import (
from sentry_sdk.utils import (
from sentry_sdk.serializer import serialize
from sentry_sdk.tracing import trace, has_tracing_enabled
from sentry_sdk.transport import HttpTransport, make_transport
from sentry_sdk.consts import (
from sentry_sdk.integrations import _DEFAULT_INTEGRATIONS, setup_integrations
from sentry_sdk.utils import ContextVar
from sentry_sdk.sessions import SessionFlusher
from sentry_sdk.envelope import Envelope
from sentry_sdk.profiler import has_profiling_enabled, Profile, setup_profiler
from sentry_sdk.scrubber import EventScrubber
from sentry_sdk.monitor import Monitor
from sentry_sdk.spotlight import setup_spotlight
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._types import TYPE_CHECKING
def _should_sample_error(self, event, hint):
    error_sampler = self.options.get('error_sampler', None)
    if callable(error_sampler):
        with capture_internal_exceptions():
            sample_rate = error_sampler(event, hint)
    else:
        sample_rate = self.options['sample_rate']
    try:
        not_in_sample_rate = sample_rate < 1.0 and random.random() >= sample_rate
    except NameError:
        logger.warning('The provided error_sampler raised an error. Defaulting to sampling the event.')
        not_in_sample_rate = False
    except TypeError:
        parameter, verb = ('error_sampler', 'returned') if callable(error_sampler) else ('sample_rate', 'contains')
        logger.warning('The provided %s %s an invalid value of %s. The value should be a float or a bool. Defaulting to sampling the event.' % (parameter, verb, repr(sample_rate)))
        not_in_sample_rate = False
    if not_in_sample_rate:
        if self.transport:
            self.transport.record_lost_event('sample_rate', data_category='error')
        return False
    return True