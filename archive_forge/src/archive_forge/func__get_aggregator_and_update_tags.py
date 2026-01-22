import io
import os
import random
import re
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from datetime import datetime
from functools import wraps, partial
import sentry_sdk
from sentry_sdk._compat import text_type, utc_from_timestamp, iteritems
from sentry_sdk.utils import (
from sentry_sdk.envelope import Envelope, Item
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
def _get_aggregator_and_update_tags(key, tags):
    hub = sentry_sdk.Hub.current
    client = hub.client
    if client is None or client.metrics_aggregator is None:
        return (None, None, tags)
    experiments = client.options.get('_experiments', {})
    updated_tags = dict(tags or ())
    updated_tags.setdefault('release', client.options['release'])
    updated_tags.setdefault('environment', client.options['environment'])
    scope = hub.scope
    local_aggregator = None
    transaction_source = scope._transaction_info.get('source')
    if transaction_source in GOOD_TRANSACTION_SOURCES:
        transaction_name = scope._transaction
        if transaction_name:
            updated_tags.setdefault('transaction', transaction_name)
        if scope._span is not None:
            sample_rate = experiments.get('metrics_summary_sample_rate')
            if sample_rate is None:
                sample_rate = 1.0
            should_summarize_metric_callback = experiments.get('should_summarize_metric')
            if random.random() < sample_rate and (should_summarize_metric_callback is None or should_summarize_metric_callback(key, updated_tags)):
                local_aggregator = scope._span._get_local_aggregator()
    before_emit_callback = experiments.get('before_emit_metric')
    if before_emit_callback is not None:
        with recursion_protection() as in_metrics:
            if not in_metrics:
                if not before_emit_callback(key, updated_tags):
                    return (None, None, updated_tags)
    return (client.metrics_aggregator, local_aggregator, updated_tags)