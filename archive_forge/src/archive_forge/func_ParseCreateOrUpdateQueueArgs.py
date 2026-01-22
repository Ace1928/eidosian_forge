from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def ParseCreateOrUpdateQueueArgs(args, queue_type, messages, is_update=False, release_track=base.ReleaseTrack.GA, http_queue=True):
    """Parses queue level args."""
    if release_track == base.ReleaseTrack.ALPHA:
        app_engine_http_target = _ParseAppEngineHttpTargetArgs(args, queue_type, messages)
        http_target = _ParseHttpTargetArgs(args, queue_type, messages) if http_queue else None
        return messages.Queue(retryConfig=_ParseRetryConfigArgs(args, queue_type, messages, is_update, is_alpha=True), rateLimits=_ParseAlphaRateLimitsArgs(args, queue_type, messages, is_update), pullTarget=_ParsePullTargetArgs(args, queue_type, messages, is_update), appEngineHttpTarget=app_engine_http_target, httpTarget=http_target)
    elif release_track == base.ReleaseTrack.BETA:
        http_target = _ParseHttpTargetArgs(args, queue_type, messages) if http_queue else None
        return messages.Queue(retryConfig=_ParseRetryConfigArgs(args, queue_type, messages, is_update, is_alpha=False), rateLimits=_ParseRateLimitsArgs(args, queue_type, messages, is_update), stackdriverLoggingConfig=_ParseStackdriverLoggingConfigArgs(args, queue_type, messages, is_update), appEngineHttpQueue=_ParseAppEngineHttpQueueArgs(args, queue_type, messages), httpTarget=http_target, type=_ParseQueueType(args, queue_type, messages, is_update))
    else:
        http_target = _ParseHttpTargetArgs(args, queue_type, messages) if http_queue else None
        return messages.Queue(retryConfig=_ParseRetryConfigArgs(args, queue_type, messages, is_update, is_alpha=False), rateLimits=_ParseRateLimitsArgs(args, queue_type, messages, is_update), stackdriverLoggingConfig=_ParseStackdriverLoggingConfigArgs(args, queue_type, messages, is_update), appEngineRoutingOverride=_ParseAppEngineRoutingOverrideArgs(args, queue_type, messages), httpTarget=http_target)