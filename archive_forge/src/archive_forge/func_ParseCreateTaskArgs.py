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
def ParseCreateTaskArgs(args, task_type, messages, release_track=base.ReleaseTrack.GA):
    """Parses task level args."""
    if release_track == base.ReleaseTrack.ALPHA:
        return messages.Task(scheduleTime=args.schedule_time, pullMessage=_ParsePullMessageArgs(args, task_type, messages), appEngineHttpRequest=_ParseAlphaAppEngineHttpRequestArgs(args, task_type, messages))
    else:
        return messages.Task(scheduleTime=args.schedule_time, appEngineHttpRequest=_ParseAppEngineHttpRequestArgs(args, task_type, messages), httpRequest=_ParseHttpRequestArgs(args, task_type, messages))