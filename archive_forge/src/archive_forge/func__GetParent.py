from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py.extra_types import _JsonValueToPythonValue
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
def _GetParent(asset):
    parent = [x.value for x in asset.resourceProperties.additionalProperties if x.key == 'name']
    if not parent:
        raise InvalidSCCInputError('No parent exists for this asset.')
    return _JsonValueToPythonValue(parent[0])