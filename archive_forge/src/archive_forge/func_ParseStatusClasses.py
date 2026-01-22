from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.monitoring import uptime
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.monitoring import flags
from googlecloudsdk.command_lib.monitoring import resource_args
from googlecloudsdk.command_lib.monitoring import util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import log
def ParseStatusClasses(status_classes):
    """Convert previously status classes from enum to flag for update logic."""
    client = uptime.UptimeClient()
    messages = client.messages
    status_mapping = {messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_1XX: '1xx', messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_2XX: '2xx', messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_3XX: '3xx', messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_4XX: '4xx', messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_5XX: '5xx', messages.ResponseStatusCode.StatusClassValueValuesEnum.STATUS_CLASS_ANY: 'any'}
    return [status_mapping.get(status_class) for status_class in status_classes]