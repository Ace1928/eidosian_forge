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
def _ParseResponse(self, response, response_collection, parameter_info=None, aggregations=None, parent_translator=None):
    """Gets a resource ref from a single item in a list response."""
    param_values = self._GetParamValuesFromParent(parameter_info, aggregations=aggregations, parent_translator=parent_translator)
    param_names = response_collection.detailed_params
    for param in param_names:
        val = getattr(response, param, None)
        if val is not None:
            param_values[param] = val
    line = getattr(response, self.id_field, '')
    return resources.REGISTRY.Parse(line, collection=response_collection.full_name, params=param_values)