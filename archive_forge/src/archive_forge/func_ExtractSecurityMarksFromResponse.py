from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py.extra_types import _JsonValueToPythonValue
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
def ExtractSecurityMarksFromResponse(response, args):
    """Returns security marks from asset response."""
    del args
    list_asset_response = list(response)
    if len(list_asset_response) > 1:
        raise InvalidSCCInputError('ListAssetResponse must only return one asset since it is filtered by Asset Name.')
    for asset_result in list_asset_response:
        return asset_result.asset.securityMarks