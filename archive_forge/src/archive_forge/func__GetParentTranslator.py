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
def _GetParentTranslator(self, parameter_info, aggregations=None):
    """Get a special parent translator if needed and available."""
    aggregations_dict = self._GetAggregationsValuesDict(aggregations)
    param_values = self._GetRawParamValuesForParent(parameter_info, aggregations_dict=aggregations_dict)
    try:
        self._ParseDefaultParent(param_values)
        return None
    except resources.ParentCollectionResolutionException:
        key = '.'.join(self._ParentParams())
        if key in _PARENT_TRANSLATORS:
            return _PARENT_TRANSLATORS.get(key)
    except resources.Error:
        return None