from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.service_extensions import flags
from googlecloudsdk.command_lib.service_extensions import util
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _ConvertStringSupportedEventToEnum(messages, supported_event):
    """Converts the text representation of an event to enum.

  Args:
    messages: module containing the definitions of messages for the API.
    supported_event: string, for example 'request_headers'.

  Returns:
    a value of messages.WasmAction.SupportedEventsValueListEntryValuesEnum,
    for example
    messages.WasmAction.SupportedEventsValueListEntryValuesEnum.REQUEST_HEADERS
  """
    uppercase_event = supported_event.upper().replace('-', '_')
    if not hasattr(messages.WasmAction.SupportedEventsValueListEntryValuesEnum, uppercase_event):
        raise ValueError('Unsupported value: ' + supported_event)
    return getattr(messages.WasmAction.SupportedEventsValueListEntryValuesEnum, uppercase_event)