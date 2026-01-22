from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py.extra_types import _JsonValueToPythonValue
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
def ExtractMatchingAssetFromDescribeResponse(response, args):
    """Returns asset that matches the user provided asset or resource-name."""
    del args
    list_asset_response = list(response)
    if not list_asset_response:
        raise InvalidSCCInputError('Asset or resource does not exist.')
    if len(list_asset_response) > 1:
        raise InvalidSCCInputError('ListAssetResponse must only return one asset since it is filtered by Asset Name or Resource Name.')
    for asset_result in list_asset_response:
        result_dictionary = {'asset': int(asset_result.asset.name.split('/')[3]), 'resourceName': asset_result.asset.securityCenterProperties.resourceName}
        return result_dictionary