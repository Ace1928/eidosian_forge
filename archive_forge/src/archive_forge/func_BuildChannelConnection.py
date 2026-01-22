from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc import common_publishing
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
def BuildChannelConnection(self, channel_connection_ref, channel, activation_token):
    return self._messages.ChannelConnection(name=channel_connection_ref.RelativeName(), channel=channel, activationToken=activation_token)