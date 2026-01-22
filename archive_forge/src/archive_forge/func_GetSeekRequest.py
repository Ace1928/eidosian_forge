from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def GetSeekRequest(args, psl):
    """Returns a SeekSubscriptionRequest from arguments."""
    if args.publish_time:
        return psl.SeekSubscriptionRequest(timeTarget=psl.TimeTarget(publishTime=util.FormatSeekTime(args.publish_time)))
    elif args.event_time:
        return psl.SeekSubscriptionRequest(timeTarget=psl.TimeTarget(eventTime=util.FormatSeekTime(args.event_time)))
    elif args.starting_offset:
        if args.starting_offset == 'beginning':
            return psl.SeekSubscriptionRequest(namedTarget=psl.SeekSubscriptionRequest.NamedTargetValueValuesEnum.TAIL)
        elif args.starting_offset == 'end':
            return psl.SeekSubscriptionRequest(namedTarget=psl.SeekSubscriptionRequest.NamedTargetValueValuesEnum.HEAD)
        else:
            raise InvalidSeekTarget('Invalid starting offset value! Must be one of: [beginning, end].')
    else:
        raise InvalidSeekTarget('Seek target must be specified!')