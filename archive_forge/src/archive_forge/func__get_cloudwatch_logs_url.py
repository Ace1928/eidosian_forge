import sys
from copy import deepcopy
from datetime import timedelta
from os import environ
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk._compat import datetime_utcnow, reraise
from sentry_sdk._types import TYPE_CHECKING
def _get_cloudwatch_logs_url(aws_context, start_time):
    """
    Generates a CloudWatchLogs console URL based on the context object

    Arguments:
        aws_context {Any} -- context from lambda handler

    Returns:
        str -- AWS Console URL to logs.
    """
    formatstring = '%Y-%m-%dT%H:%M:%SZ'
    region = environ.get('AWS_REGION', '')
    url = 'https://console.{domain}/cloudwatch/home?region={region}#logEventViewer:group={log_group};stream={log_stream};start={start_time};end={end_time}'.format(domain='amazonaws.cn' if region.startswith('cn-') else 'aws.amazon.com', region=region, log_group=aws_context.log_group_name, log_stream=aws_context.log_stream_name, start_time=(start_time - timedelta(seconds=1)).strftime(formatstring), end_time=(datetime_utcnow() + timedelta(seconds=2)).strftime(formatstring))
    return url