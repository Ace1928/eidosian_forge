from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.util import resource as resource_lib  # pylint: disable=unused-import
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.concepts import resource_parameter_info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def MessageResourceMap(self, message, ref):
    """Get dict for translating parent params into the given message type."""
    message_resource_map = {}
    for orig_param, special_param in six.iteritems(self.param_translation):
        try:
            message.field_by_name(orig_param)
        except KeyError:
            continue
        message_resource_map[orig_param] = getattr(ref, special_param, None)
    return message_resource_map