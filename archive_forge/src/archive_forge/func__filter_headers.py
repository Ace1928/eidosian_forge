from __future__ import absolute_import
import json
from copy import deepcopy
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import AnnotatedValue
from sentry_sdk._compat import text_type, iteritems
from sentry_sdk._types import TYPE_CHECKING
def _filter_headers(headers):
    if _should_send_default_pii():
        return headers
    return {k: v if k.upper().replace('-', '_') not in SENSITIVE_HEADERS else AnnotatedValue.removed_because_over_size_limit() for k, v in iteritems(headers)}