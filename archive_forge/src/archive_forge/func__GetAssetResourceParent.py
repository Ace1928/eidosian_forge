from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py.extra_types import _JsonValueToPythonValue
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
def _GetAssetResourceParent(asset_result):
    asset_parent = asset_result.asset.securityCenterProperties.resourceParent
    if asset_parent is None:
        raise InvalidSCCInputError('Asset does not have a parent.')
    return asset_parent