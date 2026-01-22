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
def _GetParamValuesFromParent(self, parameter_info, aggregations=None, parent_translator=None):
    parent_ref = self.GetParent(parameter_info, aggregations=aggregations, parent_translator=parent_translator)
    if not parent_ref:
        return {}
    params = parent_ref.AsDict()
    if parent_translator:
        return parent_translator.ToChildParams(params)
    return params