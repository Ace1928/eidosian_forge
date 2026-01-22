from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import platform
import re
import time
import uuid
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
def _LogRequest(request):
    """Logs a request."""
    uri = request.uri
    method = request.method
    headers = request.headers
    body = request.body or ''
    redact_req_body_reason = None
    redact_resp_body_reason = None
    if redact_token and IsTokenUri(uri):
        redact_req_body_reason = 'Contains oauth token. Set log_http_redact_token property to false to print the body of this request.'
        redact_resp_body_reason = 'Contains oauth token. Set log_http_redact_token property to false to print the body of this response.'
    elif redact_request_body_reason is not None:
        redact_req_body_reason = redact_request_body_reason
    log.status.Print('=======================')
    log.status.Print('==== request start ====')
    log.status.Print('uri: {uri}'.format(uri=uri))
    log.status.Print('method: {method}'.format(method=method))
    log.status.Print('== headers start ==')
    for h, v in sorted(six.iteritems(headers)):
        if redact_token and h.lower() in (b'authorization', b'x-goog-iam-authorization-token'):
            v = '--- Token Redacted ---'
        log.status.Print('{0}: {1}'.format(h, v))
    log.status.Print('== headers end ==')
    log.status.Print('== body start ==')
    if redact_req_body_reason is None:
        log.status.Print(body)
    else:
        log.status.Print('Body redacted: {}'.format(redact_req_body_reason))
    log.status.Print('== body end ==')
    log.status.Print('==== request end ====')
    return {'start_time': time.time(), 'redact_resp_body_reason': redact_resp_body_reason}