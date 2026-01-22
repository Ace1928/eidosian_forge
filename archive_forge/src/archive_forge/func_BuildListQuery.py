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
def BuildListQuery(self, parameter_info, aggregations=None, parent_translator=None):
    """Builds a list request to list values for the given argument.

    Args:
      parameter_info: the runtime ResourceParameterInfo object.
      aggregations: a list of _RuntimeParameter objects.
      parent_translator: a ParentTranslator object if needed.

    Returns:
      The apitools request.
    """
    method = self.method
    if method is None:
        return None
    message = method.GetRequestType()()
    for field, value in six.iteritems(self._static_params):
        arg_utils.SetFieldInMessage(message, field, value)
    parent = self.GetParent(parameter_info, aggregations=aggregations, parent_translator=parent_translator)
    if not parent:
        return message
    message_resource_map = {}
    if parent_translator:
        message_resource_map = parent_translator.MessageResourceMap(message, parent)
    arg_utils.ParseResourceIntoMessage(parent, method, message, message_resource_map=message_resource_map, is_primary_resource=True)
    return message