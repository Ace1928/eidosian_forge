from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _GetMessageFormatString(pubsub_config):
    message_format_type = getattr(pubsub_config, 'messageFormat')
    message_format_enums = _MESSAGES.PubsubConfig.MessageFormatValueValuesEnum
    if message_format_type == message_format_enums.PROTOBUF:
        return 'protobuf'
    return 'json'