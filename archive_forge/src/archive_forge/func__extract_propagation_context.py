from copy import copy
from collections import deque
from itertools import chain
import os
import sys
import uuid
from sentry_sdk.attachments import Attachment
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk.consts import FALSE_VALUES, INSTRUMENTER
from sentry_sdk._functools import wraps
from sentry_sdk.profiler import Profile
from sentry_sdk.session import Session
from sentry_sdk.tracing_utils import (
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def _extract_propagation_context(self, data):
    context = {}
    normalized_data = normalize_incoming_data(data)
    baggage_header = normalized_data.get(BAGGAGE_HEADER_NAME)
    if baggage_header:
        context['dynamic_sampling_context'] = Baggage.from_incoming_header(baggage_header).dynamic_sampling_context()
    sentry_trace_header = normalized_data.get(SENTRY_TRACE_HEADER_NAME)
    if sentry_trace_header:
        sentrytrace_data = extract_sentrytrace_data(sentry_trace_header)
        if sentrytrace_data is not None:
            context.update(sentrytrace_data)
    only_baggage_no_sentry_trace = 'dynamic_sampling_context' in context and 'trace_id' not in context
    if only_baggage_no_sentry_trace:
        context.update(self._create_new_propagation_context())
    if context:
        if not context.get('span_id'):
            context['span_id'] = uuid.uuid4().hex[16:]
        return context
    return None